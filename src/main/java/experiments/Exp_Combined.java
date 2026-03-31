package experiments;

import algorithms.PTK_HUIM_DFS;
import algorithms.PTK_HUIM_BFS;
import algorithms.PTK_HUIM_BestFS;
import algorithms.UTKU_PSO;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Exp_Combined: Runs both Accuracy and Memory experiments in a single pass with
 * multiple runs for statistical validity (IEEE Access compliant).
 *
 * Protocol: 1 warmup + N measured runs per (dataset, k, algorithm)
 *   - Deterministic algorithms (PTK_DFS/BFS/BestFS): 3 measured runs
 *   - Stochastic algorithm (UTKU_PSO): 5 measured runs
 *
 * For each (dataset, k):
 *   - Runs PTK-HUIM-DFS N times → collects memory stats + ground truth for accuracy
 *   - Runs PTK-HUIM-BFS N times → collects memory stats
 *   - Runs PTK-HUIM-BestFS N times → collects memory stats
 *   - Runs UTKU-PSO M times → collects memory stats + accuracy comparison
 *
 * Outputs:
 *   - accuracy_<timestamp>.txt (mean ± stddev accuracy)
 *   - memory_<timestamp>.txt (mean ± stddev memory, peak memory)
 *
 * Usage: java experiments.Exp_Combined [--output-dir results/]
 */
public class Exp_Combined {

    // =========================================================================
    // Experimental Protocol Constants
    // =========================================================================
    private static final int WARMUP_RUNS = 1;
    private static final int MEASURED_RUNS_DETERMINISTIC = 3;  // PTK algorithms
    private static final int MEASURED_RUNS_STOCHASTIC = 3;     // UTKU_PSO

    private static final int PSO_POP_SIZE = 20;
    private static final int PSO_ITERATIONS = 10000;

    private static final String[][] DATASETS = {
        {"Chess",    "src/data/chess_database.txt",    "src/data/chess_profit.txt"},
        //{"Mushroom", "src/data/mushroom_database.txt", "src/data/mushroom_profit.txt"},
        //{"Accidents",  "src/data/accidents_database.txt",  "src/data/accidents_profit.txt"},
        //{"Retail",   "src/data/retail_database.txt",   "src/data/retail_profit.txt"},
        //{"Kosarak",  "src/data/kosarak_database.txt",  "src/data/kosarak_profit.txt"},
        //{"Pumsb",  "src/data/pumsb_database.txt",  "src/data/pumsb_profit.txt"},
    };

    private static final Map<String, int[]> K_VALUES = new LinkedHashMap<>();
    static {
        K_VALUES.put("Chess",    new int[]{1, 5, 10, 50});//, 5000, 10000, 20000});
        K_VALUES.put("Mushroom", new int[]{10, 100, 1000});//, 5000, 10000, 20000});
        K_VALUES.put("Retail",   new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Accidents",  new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Kosarak",  new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Pumsb",  new int[]{1, 10, 50, 100, 150, 200});
    }

    private static final String[] ALL_ALGOS = {"PTK_DFS", "PTK_BFS", "PTK_BestFS", "UTKU_PSO"};

    // =========================================================================
    // Result containers
    // =========================================================================
    static class AccuracyResult {
        String dataset;
        int k;
        int exactCount;
        int heuristicCount;
        double meanAccuracyPct;
        double stddevAccuracy;
        double minAccuracyPct;
        double maxAccuracyPct;
        long exactTimeMs;
        long heuristicTimeMs;
        int runs;
        double[] accuracyRuns;  // Individual run accuracies
        boolean exactFailed;
        String failReason;
    }

    static class MemoryResult {
        String algorithm;
        String dataset;
        int k;
        double meanMemoryMB;
        double stddevMemoryMB;
        double peakMemoryMB;  // Maximum across all runs
        int patternCount;
        long meanRuntimeMs;
        int runs;
        double[] memoryRuns;  // Individual run memories
        List<Set<String>> allPatterns = new ArrayList<>();  // Patterns from each run (for accuracy)
        boolean failed;
        String failReason;
    }

    // Single run result
    static class AlgoRunResult {
        Set<String> patterns = new LinkedHashSet<>();
        int patternCount;
        double peakMemoryMB;
        long runtimeMs;
        boolean failed;
        String failReason;
    }

    // =========================================================================
    // Main
    // =========================================================================
    public static void main(String[] args) throws Exception {
        String outputDir = "results/";

        for (String arg : args) {
            if (arg.startsWith("--output-dir=")) outputDir = arg.substring(13);
        }

        new File(outputDir).mkdirs();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String accuracyFile = outputDir + "accuracy_" + timestamp + ".txt";
        String memoryFile = outputDir + "memory_" + timestamp + ".txt";

        List<AccuracyResult> accuracyResults = new ArrayList<>();
        List<MemoryResult> memoryResults = new ArrayList<>();
        int totalCorrect = 0, totalK = 0;

        // Open both output writers
        try (PrintWriter accW = new PrintWriter(new BufferedWriter(new FileWriter(accuracyFile)));
             PrintWriter memW = new PrintWriter(new BufferedWriter(new FileWriter(memoryFile)))) {

            writeAccuracyHeader(accW);
            writeMemoryHeader(memW);

            for (String[] ds : DATASETS) {
                String name = ds[0], db = ds[1], prof = ds[2];
                int[] kVals = K_VALUES.get(name);
                if (kVals == null) continue;

                if (!new File(db).exists() || !new File(prof).exists()) {
                    String skipMsg = String.format(">>> SKIPPING %s: data files not found", name);
                    accW.printf("%n%s%n", skipMsg);
                    memW.printf("%n%s%n", skipMsg);
                    System.out.println(skipMsg);
                    continue;
                }

                // Dataset header
                String dsHeader = String.format("%n%s%nDATASET: %s%n%s",
                    "=".repeat(80), name, "=".repeat(80));
                accW.println(dsHeader);
                memW.println(dsHeader);
                System.out.printf("%n=== DATASET: %s ===%n", name);

                for (int k : kVals) {
                    System.out.printf("  k = %d%n", k);
                    memW.printf("%n--- k = %d ---%n", k);

                    // Store results for this k
                    Map<String, MemoryResult> memResults = new LinkedHashMap<>();

                    // Run all algorithms with multiple runs
                    for (String algoName : ALL_ALGOS) {
                        System.out.printf("    [%s] ", algoName);
                        System.out.flush();

                        MemoryResult mr = runAlgorithmMultiple(algoName, db, prof, k);
                        mr.dataset = name;
                        mr.k = k;
                        memResults.put(algoName, mr);
                        memoryResults.add(mr);

                        if (mr.failed) {
                            System.out.printf("FAILED (%s)%n", mr.failReason);
                        } else {
                            System.out.printf(Locale.US, "mean=%.2f MB (±%.2f), peak=%.2f MB%n",
                                    mr.meanMemoryMB, mr.stddevMemoryMB, mr.peakMemoryMB);
                        }

                        // Aggressive GC between algorithms
                        System.gc();
                        sleep(300);
                    }

                    // Process accuracy comparison (PTK_DFS vs UTKU_PSO)
                    AccuracyResult ar = calculateAccuracy(name, k, memResults, accW);
                    accuracyResults.add(ar);

                    if (!ar.exactFailed) {
                        totalCorrect += (int)(ar.meanAccuracyPct * ar.exactCount / 100.0);
                        totalK += ar.exactCount;
                    }

                    // Write memory results for this k
                    writeMemoryRow(memW, memResults);

                    System.gc();
                    sleep(300);
                }
            }

            // Write accuracy summary
            accW.printf("%n%n%s%n", "=".repeat(80));
            accW.println("ACCURACY SUMMARY TABLE (for paper)");
            accW.printf("%s%n%n", "=".repeat(80));
            writeAccuracyTable(accW, accuracyResults);

            double overallAccuracy = totalK > 0 ? (double) totalCorrect / totalK * 100.0 : 0;
            accW.printf(Locale.US, "%nOverall mean accuracy: %.1f%%%n", overallAccuracy);
            accW.printf("%n%s%n", "=".repeat(80));

            // Write memory summary
            memW.printf("%n%n%s%n", "=".repeat(80));
            memW.println("MEMORY SUMMARY TABLE (for paper figures)");
            memW.printf("%s%n", "=".repeat(80));
            writeMemorySummary(memW, memoryResults);
        }

        System.out.printf("%nResults saved to:%n  - %s%n  - %s%n", accuracyFile, memoryFile);
    }

    // =========================================================================
    // Multi-run algorithm execution
    // =========================================================================

    /**
     * Runs an algorithm multiple times and computes statistics.
     */
    private static MemoryResult runAlgorithmMultiple(String algoName, String db, String prof, int k) {
        MemoryResult mr = new MemoryResult();
        mr.algorithm = algoName;

        int measuredRuns = algoName.equals("UTKU_PSO") ?
                MEASURED_RUNS_STOCHASTIC : MEASURED_RUNS_DETERMINISTIC;

        // Warmup runs
        System.out.printf("warmup ");
        for (int i = 0; i < WARMUP_RUNS; i++) {
            try {
                runAlgorithmOnce(algoName, db, prof, k);
                System.out.printf(".");
            } catch (Exception e) {
                // Ignore warmup failures
            }
            System.gc();
            sleep(200);
        }

        // Measured runs
        System.out.printf(" measure ");
        List<AlgoRunResult> results = new ArrayList<>();
        for (int i = 0; i < measuredRuns; i++) {
            System.out.printf(".");
            System.out.flush();

            AlgoRunResult result = runAlgorithmOnce(algoName, db, prof, k);
            if (result.failed) {
                mr.failed = true;
                mr.failReason = result.failReason;
                return mr;
            }
            results.add(result);

            System.gc();
            sleep(300);
        }
        System.out.printf(" ");

        // Compute statistics
        mr.runs = results.size();
        mr.memoryRuns = new double[mr.runs];

        double sumMemory = 0, sumTime = 0;
        mr.peakMemoryMB = 0;
        mr.patternCount = results.get(0).patternCount;

        for (int i = 0; i < results.size(); i++) {
            AlgoRunResult r = results.get(i);
            mr.memoryRuns[i] = r.peakMemoryMB;
            sumMemory += r.peakMemoryMB;
            sumTime += r.runtimeMs;
            mr.peakMemoryMB = Math.max(mr.peakMemoryMB, r.peakMemoryMB);

            // Store patterns from each run for accuracy calculation
            if (!r.patterns.isEmpty()) {
                mr.allPatterns.add(r.patterns);
            }
        }

        mr.meanMemoryMB = sumMemory / mr.runs;
        mr.meanRuntimeMs = (long)(sumTime / mr.runs);

        // Calculate stddev
        double variance = 0;
        for (double mem : mr.memoryRuns) {
            variance += Math.pow(mem - mr.meanMemoryMB, 2);
        }
        mr.stddevMemoryMB = Math.sqrt(variance / mr.runs);

        return mr;
    }

    /**
     * Runs algorithm once and returns raw result.
     */
    private static AlgoRunResult runAlgorithmOnce(String algoName, String db, String prof, int k) {
        AlgoRunResult result = new AlgoRunResult();

        // Force GC before measurement
        System.gc();
        sleep(100);

        String tmpOut = "tmp_combined_" + algoName + "_" + System.nanoTime() + ".txt";
        try {
            long start = System.currentTimeMillis();

            switch (algoName) {
                case "PTK_DFS": {
                    PTK_HUIM_DFS algo = new PTK_HUIM_DFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_DFS.MiningResult r = algo.run();
                    result.peakMemoryMB = r.memoryUsedMB;
                    result.patternCount = r.patterns.size();
                    // Extract patterns for accuracy comparison
                    for (PTK_HUIM_DFS.Pattern p : r.patterns) {
                        result.patterns.add(patternKey(p.items));
                    }
                    break;
                }
                case "PTK_BFS": {
                    PTK_HUIM_BFS algo = new PTK_HUIM_BFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_BFS.MiningResult r = algo.run();
                    result.peakMemoryMB = r.memoryUsedMB;
                    result.patternCount = r.patterns.size();
                    break;
                }
                case "PTK_BestFS": {
                    PTK_HUIM_BestFS algo = new PTK_HUIM_BestFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_BestFS.MiningResult r = algo.run();
                    result.peakMemoryMB = r.memoryUsedMB;
                    result.patternCount = r.patterns.size();
                    break;
                }
                case "UTKU_PSO": {
                    UTKU_PSO algo = new UTKU_PSO(db, prof, tmpOut, k, PSO_POP_SIZE, PSO_ITERATIONS);
                    UTKU_PSO.MiningResult r = algo.run();
                    result.peakMemoryMB = r.memoryUsedMB;
                    result.patternCount = r.patternCount;
                    // Parse patterns for accuracy comparison
                    result.patterns = parsePSOPatterns(tmpOut);
                    break;
                }
            }

            result.runtimeMs = System.currentTimeMillis() - start;
        } catch (Exception e) {
            result.failed = true;
            result.failReason = e.getClass().getSimpleName() + ": " + e.getMessage();
        } finally {
            new File(tmpOut).delete();
        }

        return result;
    }

    // =========================================================================
    // Accuracy calculation with multiple runs
    // =========================================================================

    /**
     * Calculates accuracy by comparing PTK_DFS and UTKU_PSO patterns from multiple runs.
     * For deterministic PTK_DFS: patterns should be identical across runs, use first run as ground truth.
     * For stochastic UTKU_PSO: calculate mean accuracy across runs.
     */
    private static AccuracyResult calculateAccuracy(String dataset, int k,
                                                      Map<String, MemoryResult> memResults,
                                                      PrintWriter accW) {
        AccuracyResult ar = new AccuracyResult();
        ar.dataset = dataset;
        ar.k = k;

        MemoryResult exactMem = memResults.get("PTK_DFS");
        MemoryResult heuristicMem = memResults.get("UTKU_PSO");

        if (exactMem.failed) {
            ar.exactFailed = true;
            ar.failReason = exactMem.failReason;
            accW.printf("  k=%-6d  EXACT FAILED: %s%n", k, exactMem.failReason);
            return ar;
        }

        if (heuristicMem.failed) {
            ar.exactFailed = true;
            ar.failReason = "HEURISTIC: " + heuristicMem.failReason;
            accW.printf("  k=%-6d  HEURISTIC FAILED: %s%n", k, heuristicMem.failReason);
            return ar;
        }

        // Check if we have patterns from runs
        if (exactMem.allPatterns.isEmpty() || heuristicMem.allPatterns.isEmpty()) {
            ar.exactFailed = true;
            ar.failReason = "No patterns available for comparison";
            accW.printf("  k=%-6d  ERROR: No patterns available%n", k);
            return ar;
        }

        // Use first PTK_DFS run as ground truth (deterministic algorithm)
        Set<String> groundTruth = exactMem.allPatterns.get(0);
        int effectiveK = Math.max(groundTruth.size(), 1);

        ar.exactCount = effectiveK;
        ar.heuristicCount = heuristicMem.patternCount;
        ar.exactTimeMs = exactMem.meanRuntimeMs;
        ar.heuristicTimeMs = heuristicMem.meanRuntimeMs;
        ar.runs = heuristicMem.runs;

        // Calculate accuracy for each UTKU_PSO run
        ar.accuracyRuns = new double[ar.runs];
        double sumAccuracy = 0;
        double minAcc = 100.0;
        double maxAcc = 0.0;

        for (int i = 0; i < heuristicMem.allPatterns.size(); i++) {
            Set<String> heuristicPatterns = heuristicMem.allPatterns.get(i);

            // Count matches
            int correct = 0;
            for (String pattern : heuristicPatterns) {
                if (groundTruth.contains(pattern)) {
                    correct++;
                }
            }

            double accuracy = (double) correct / effectiveK * 100.0;
            ar.accuracyRuns[i] = accuracy;
            sumAccuracy += accuracy;
            minAcc = Math.min(minAcc, accuracy);
            maxAcc = Math.max(maxAcc, accuracy);
        }

        ar.meanAccuracyPct = sumAccuracy / ar.runs;
        ar.minAccuracyPct = minAcc;
        ar.maxAccuracyPct = maxAcc;

        // Calculate stddev
        double variance = 0;
        for (double acc : ar.accuracyRuns) {
            variance += Math.pow(acc - ar.meanAccuracyPct, 2);
        }
        ar.stddevAccuracy = Math.sqrt(variance / ar.runs);

        accW.printf("  k=%-6d  exact=%d  heuristic=%d%n", k, ar.exactCount, ar.heuristicCount);
        accW.printf(Locale.US, "           mean accuracy=%.1f%% (±%.1f%%)  range=[%.1f%%, %.1f%%]%n",
                ar.meanAccuracyPct, ar.stddevAccuracy, ar.minAccuracyPct, ar.maxAccuracyPct);

        // Print individual run accuracies for reference
        accW.printf("           runs: ");
        for (int i = 0; i < ar.accuracyRuns.length; i++) {
            accW.printf(Locale.US, "%.1f%%", ar.accuracyRuns[i]);
            if (i < ar.accuracyRuns.length - 1) accW.printf(", ");
        }
        accW.printf("%n");

        return ar;
    }

    /**
     * Parses UTKU-PSO output file to extract pattern itemsets as string keys.
     */
    static Set<String> parsePSOPatterns(String outputFile) throws IOException {
        Set<String> patterns = new LinkedHashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(outputFile))) {
            String line;
            boolean inResults = false;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("Rank") && line.contains("Pattern")) {
                    inResults = true;
                    continue;
                }
                if (line.startsWith("---") || line.startsWith("===")) {
                    if (inResults && line.startsWith("===")) break;
                    continue;
                }
                if (!inResults || line.isEmpty()) continue;

                // Parse pattern: "1      {3, 5, 7}   ..."
                int braceStart = line.indexOf('{');
                int braceEnd = line.indexOf('}');
                if (braceStart >= 0 && braceEnd > braceStart) {
                    String itemStr = line.substring(braceStart + 1, braceEnd);
                    List<Integer> items = new ArrayList<>();
                    for (String s : itemStr.split(",")) {
                        s = s.trim();
                        if (!s.isEmpty()) {
                            try { items.add(Integer.parseInt(s)); }
                            catch (NumberFormatException ignored) {}
                        }
                    }
                    Collections.sort(items);
                    patterns.add(items.stream().map(String::valueOf).collect(Collectors.joining(",")));
                }
            }
        }
        return patterns;
    }

    /**
     * Creates a canonical string key from a set of item IDs.
     */
    private static String patternKey(Set<Integer> items) {
        List<Integer> sorted = new ArrayList<>(items);
        Collections.sort(sorted);
        return sorted.stream().map(String::valueOf).collect(Collectors.joining(","));
    }

    // =========================================================================
    // Output formatting
    // =========================================================================
    private static void writeAccuracyHeader(PrintWriter w) {
        w.printf("%s%n", "=".repeat(80));
        w.println("ACCURACY COMPARISON BENCHMARK (IEEE Access Section 6.C)");
        w.printf("%s%n", "=".repeat(80));
        w.printf("  Ground Truth:    PTK-HUIM-DFS (exact, %d runs)%n", MEASURED_RUNS_DETERMINISTIC);
        w.printf("  Heuristic:       UTKU-PSO (pop=%d, iter=%d, %d runs)%n",
                PSO_POP_SIZE, PSO_ITERATIONS, MEASURED_RUNS_STOCHASTIC);
        w.printf("  Formula:         accuracy = |correct| / |exact| × 100%%%n");
        w.printf("  Matching:        by itemset identity (sorted item IDs)%n");
        w.printf("  Statistics:      mean ± stddev across multiple runs%n");
        w.printf("  Timestamp:       %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        w.printf("%s%n", "=".repeat(80));
    }

    private static void writeMemoryHeader(PrintWriter w) {
        w.printf("%s%n", "=".repeat(80));
        w.println("MEMORY USAGE BENCHMARK (IEEE Access Section 6.D)");
        w.printf("%s%n", "=".repeat(80));
        w.printf("  Measurement:     Runtime.totalMemory() - Runtime.freeMemory()%n");
        w.printf("  Sampling:        After each major phase (peak captured)%n");
        w.printf("  Warmup Runs:     %d%n", WARMUP_RUNS);
        w.printf("  Measured Runs:   %d (deterministic), %d (stochastic)%n",
                MEASURED_RUNS_DETERMINISTIC, MEASURED_RUNS_STOCHASTIC);
        w.printf("  Statistics:      mean ± stddev, peak (max across runs)%n");
        w.printf("  JDK:             %s%n", System.getProperty("java.version"));
        w.printf("  Max Heap:        %d MB%n", Runtime.getRuntime().maxMemory() / (1024 * 1024));
        w.printf("  Timestamp:       %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        w.printf("%s%n", "=".repeat(80));
    }

    private static void writeAccuracyTable(PrintWriter w, List<AccuracyResult> results) {
        // Group by dataset
        Map<String, List<AccuracyResult>> byDataset = new LinkedHashMap<>();
        for (AccuracyResult r : results) {
            byDataset.computeIfAbsent(r.dataset, x -> new ArrayList<>()).add(r);
        }

        // Collect all unique k values for header
        Set<Integer> allKs = new LinkedHashSet<>();
        for (AccuracyResult r : results) allKs.add(r.k);
        List<Integer> kList = new ArrayList<>(allKs);

        // Print summary table (mean accuracy)
        w.println("  Summary Table (Mean Accuracy %):");
        StringBuilder header = new StringBuilder(String.format("  %-12s", "Dataset"));
        for (int k : kList) header.append(String.format(" %12s", "k=" + k));
        w.println(header);
        w.printf("  %s%n", "-".repeat(12 + kList.size() * 13));

        // Print rows (mean accuracy)
        for (Map.Entry<String, List<AccuracyResult>> entry : byDataset.entrySet()) {
            StringBuilder row = new StringBuilder(String.format("  %-12s", entry.getKey()));
            Map<Integer, AccuracyResult> byK = new LinkedHashMap<>();
            for (AccuracyResult r : entry.getValue()) byK.put(r.k, r);

            for (int k : kList) {
                AccuracyResult r = byK.get(k);
                if (r == null) {
                    row.append(String.format(" %12s", "-"));
                } else if (r.exactFailed) {
                    row.append(String.format(" %12s", "DNF"));
                } else {
                    row.append(String.format(Locale.US, " %11.1f%%", r.meanAccuracyPct));
                }
            }
            w.println(row);
        }

        // Write detailed run-by-run accuracy data
        w.printf("%n%n  Detailed Run-by-Run Accuracy Data:%n");
        w.printf("  %s%n", "=".repeat(78));

        for (Map.Entry<String, List<AccuracyResult>> entry : byDataset.entrySet()) {
            w.printf("%n  Dataset: %s%n", entry.getKey());

            for (AccuracyResult r : entry.getValue()) {
                if (r.exactFailed) {
                    w.printf("    k=%-6d: FAILED - %s%n", r.k, r.failReason);
                } else {
                    w.printf("    k=%-6d: ", r.k);
                    if (r.accuracyRuns != null) {
                        for (int i = 0; i < r.accuracyRuns.length; i++) {
                            w.printf(Locale.US, "Run%d=%.1f%%", i + 1, r.accuracyRuns[i]);
                            if (i < r.accuracyRuns.length - 1) w.printf(", ");
                        }
                        w.printf(Locale.US, "  | Mean=%.1f%%, StdDev=%.1f%%, Range=[%.1f%%-%.1f%%]%n",
                                r.meanAccuracyPct, r.stddevAccuracy, r.minAccuracyPct, r.maxAccuracyPct);
                    }
                }
            }
        }
    }

    private static void writeMemoryRow(PrintWriter w, Map<String, MemoryResult> results) {
        w.printf("  %-15s %8s %15s %15s %10s  %s%n",
                "Algorithm", "Patterns", "Mean(MB)", "StdDev(MB)", "Peak(MB)", "Individual Runs (MB)");
        w.printf("  %s%n", "-".repeat(120));
        for (String algo : ALL_ALGOS) {
            MemoryResult r = results.get(algo);
            if (r.failed) {
                w.printf("  %-15s %8s %15s  %s%n", algo, "ERR", "FAILED", r.failReason);
            } else {
                w.printf(Locale.US, "  %-15s %8d %15.2f %15.2f %10.2f  [",
                        algo, r.patternCount, r.meanMemoryMB, r.stddevMemoryMB, r.peakMemoryMB);

                // Write individual run values
                for (int i = 0; i < r.memoryRuns.length; i++) {
                    w.printf(Locale.US, "%.2f", r.memoryRuns[i]);
                    if (i < r.memoryRuns.length - 1) w.printf(", ");
                }
                w.printf("]%n");
            }
        }
    }

    private static void writeMemorySummary(PrintWriter w, List<MemoryResult> results) {
        // Group by dataset
        Map<String, Map<Integer, Map<String, MemoryResult>>> grouped = new LinkedHashMap<>();
        for (MemoryResult r : results) {
            grouped.computeIfAbsent(r.dataset, x -> new LinkedHashMap<>())
                   .computeIfAbsent(r.k, x -> new LinkedHashMap<>())
                   .put(r.algorithm, r);
        }

        // First: Write summary table (for paper)
        for (Map.Entry<String, Map<Integer, Map<String, MemoryResult>>> dsEntry : grouped.entrySet()) {
            w.printf("%n  Dataset: %s (Mean Memory ± StdDev MB)%n", dsEntry.getKey());
            w.printf("  %-8s", "k");
            for (String algo : ALL_ALGOS) w.printf(" %18s", algo);
            w.println();
            w.printf("  %s%n", "-".repeat(8 + ALL_ALGOS.length * 19));

            for (Map.Entry<Integer, Map<String, MemoryResult>> kEntry : dsEntry.getValue().entrySet()) {
                w.printf("  %-8d", kEntry.getKey());
                for (String algo : ALL_ALGOS) {
                    MemoryResult r = kEntry.getValue().get(algo);
                    if (r == null || r.failed) {
                        w.printf(" %18s", "DNF");
                    } else {
                        w.printf(Locale.US, " %7.2f (±%6.2f)", r.meanMemoryMB, r.stddevMemoryMB);
                    }
                }
                w.println();
            }
        }

        // Second: Write detailed run-by-run data
        w.printf("%n%n%s%n", "=".repeat(80));
        w.println("DETAILED RUN-BY-RUN MEMORY DATA");
        w.printf("%s%n", "=".repeat(80));

        for (Map.Entry<String, Map<Integer, Map<String, MemoryResult>>> dsEntry : grouped.entrySet()) {
            w.printf("%n  Dataset: %s%n", dsEntry.getKey());

            for (Map.Entry<Integer, Map<String, MemoryResult>> kEntry : dsEntry.getValue().entrySet()) {
                w.printf("%n    k = %d:%n", kEntry.getKey());

                for (String algo : ALL_ALGOS) {
                    MemoryResult r = kEntry.getValue().get(algo);
                    if (r == null || r.failed) {
                        w.printf("      %-15s: FAILED%n", algo);
                    } else {
                        w.printf("      %-15s: ", algo);
                        for (int i = 0; i < r.memoryRuns.length; i++) {
                            w.printf(Locale.US, "Run%d=%.2f MB", i + 1, r.memoryRuns[i]);
                            if (i < r.memoryRuns.length - 1) w.printf(", ");
                        }
                        w.printf(Locale.US, "  | Mean=%.2f, Peak=%.2f%n",
                                r.meanMemoryMB, r.peakMemoryMB);
                    }
                }
            }
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}
