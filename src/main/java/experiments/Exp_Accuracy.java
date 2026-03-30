package experiments;

import algorithms.PTK_HUIM_DFS;
import algorithms.UTKU_PSO;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Exp_Accuracy: Measures accuracy of UTKU-PSO against PTK-HUIM-DFS ground truth.
 *
 * IEEE Access Section 6.C — generates data for accuracy table.
 *
 * For each (dataset, k):
 *   1. Run PTK-HUIM-DFS to get exact top-k patterns (ground truth)
 *   2. Run UTKU-PSO to get approximate top-k patterns
 *   3. Accuracy = |intersection| / k × 100%
 *
 * Matching is by itemset identity (sorted item IDs), not by EU value.
 *
 * Usage: java algorithms.Exp_Accuracy [--output-dir results/] [--min-prob 0.1]
 */
public class Exp_Accuracy {
    private static final int PSO_POP_SIZE = 20;
    private static final int PSO_ITERATIONS = 10000;

    private static final String[][] DATASETS = {
        {"Chess",    "src/data/chess_database.txt",    "src/data/chess_profit.txt"},
        {"Mushroom", "src/data/mushroom_database.txt", "src/data/mushroom_profit.txt"},
        {"Accidents",  "src/data/accidents_database.txt",  "src/data/accidents_profit.txt"},
        {"Retail",   "src/data/retail_database.txt",   "src/data/retail_profit.txt"},
        {"Kosarak",  "src/data/kosarak_database.txt",  "src/data/kosarak_profit.txt"},
        {"Pumsb",  "src/data/pumsb_database.txt",  "src/data/pumsb_profit.txt"},
    };

    private static final Map<String, int[]> K_VALUES = new LinkedHashMap<>();
    static {
        K_VALUES.put("Chess",    new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Mushroom", new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Retail",   new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Accidents",  new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Kosarak",  new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Pumsb",  new int[]{1, 10, 50, 100, 150, 200});
    }

    // =========================================================================
    // Result container
    // =========================================================================
    static class AccuracyResult {
        String dataset;
        int k;
        int exactCount;
        int heuristicCount;
        int correctCount;
        double accuracyPct;
        long exactTimeMs;
        long heuristicTimeMs;
        boolean exactFailed;
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
        String outFile = outputDir + "accuracy_" + timestamp + ".txt";

        List<AccuracyResult> allResults = new ArrayList<>();
        int totalCorrect = 0, totalK = 0;

        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(outFile)))) {
            writeHeader(w);

            for (String[] ds : DATASETS) {
                String name = ds[0], db = ds[1], prof = ds[2];
                int[] kVals = K_VALUES.get(name);
                if (kVals == null) continue;

                if (!new File(db).exists() || !new File(prof).exists()) {
                    w.printf("%n>>> SKIPPING %s: data files not found%n", name);
                    System.out.printf(">>> SKIPPING %s: data files not found%n", name);
                    continue;
                }

                w.printf("%n%s%n", "=".repeat(80));
                w.printf("DATASET: %s%n", name);
                w.printf("%s%n", "=".repeat(80));
                System.out.printf("%n=== DATASET: %s ===%n", name);

                for (int k : kVals) {
                    System.out.printf("  k = %d ... ", k);
                    System.out.flush();

                    AccuracyResult ar = new AccuracyResult();
                    ar.dataset = name;
                    ar.k = k;

                    // Step 1: Run exact algorithm (ground truth)
                    Set<String> groundTruth;
                    try {
                        long t0 = System.currentTimeMillis();
                        groundTruth = runExact(db, prof, k);
                        ar.exactTimeMs = System.currentTimeMillis() - t0;
                        ar.exactCount = groundTruth.size();
                    } catch (Exception e) {
                        ar.exactFailed = true;
                        ar.failReason = e.getMessage();
                        w.printf("  k=%-6d  EXACT FAILED: %s%n", k, e.getMessage());
                        System.out.printf("EXACT FAILED%n");
                        allResults.add(ar);
                        continue;
                    }

                    // Step 2: Run heuristic algorithm
                    Set<String> heuristicResult;
                    try {
                        long t0 = System.currentTimeMillis();
                        heuristicResult = runHeuristic(db, prof, k);
                        ar.heuristicTimeMs = System.currentTimeMillis() - t0;
                        ar.heuristicCount = heuristicResult.size();
                    } catch (Exception e) {
                        ar.exactFailed = true;
                        ar.failReason = "HEURISTIC: " + e.getMessage();
                        w.printf("  k=%-6d  HEURISTIC FAILED: %s%n", k, e.getMessage());
                        System.out.printf("HEURISTIC FAILED%n");
                        allResults.add(ar);
                        continue;
                    }

                    // Step 3: Compare
                    int correct = 0;
                    for (String pattern : heuristicResult) {
                        if (groundTruth.contains(pattern)) correct++;
                    }
                    int effectiveK = Math.max(ar.exactCount, 1);
                    ar.correctCount = correct;
                    ar.accuracyPct = (double) correct / effectiveK * 100.0;

                    totalCorrect += correct;
                    totalK += effectiveK;
                    allResults.add(ar);

                    w.printf(Locale.US, "  k=%-6d  exact=%d  heuristic=%d  correct=%d  accuracy=%.1f%%  (exact %dms, heuristic %dms)%n",
                            k, ar.exactCount, ar.heuristicCount, correct, ar.accuracyPct,
                            ar.exactTimeMs, ar.heuristicTimeMs);
                    System.out.printf(Locale.US, "accuracy=%.1f%%%n", ar.accuracyPct);

                    System.gc();
                    sleep(200);
                }
            }

            // Summary table (IEEE Access TABLE format)
            w.printf("%n%n%s%n", "=".repeat(80));
            w.println("ACCURACY SUMMARY TABLE (for paper)");
            w.printf("%s%n%n", "=".repeat(80));

            writeAccuracyTable(w, allResults);

            // Overall
            double overallAccuracy = totalK > 0 ? (double) totalCorrect / totalK * 100.0 : 0;
            w.printf(Locale.US, "%nOverall accuracy: %d / %d = %.1f%%%n", totalCorrect, totalK, overallAccuracy);
            w.printf("%n%s%n", "=".repeat(80));
        }

        System.out.printf("%nResults saved to: %s%n", outFile);
    }

    // =========================================================================
    // Algorithm Runners
    // =========================================================================

    /**
     * Runs PTK-HUIM-DFS and returns a set of pattern keys for comparison.
     * Pattern key = sorted item IDs joined by comma, e.g., "1,3,5"
     */
    private static Set<String> runExact(String db, String prof, int k) throws Exception {
        String tmpOut = "tmp_exact_" + Thread.currentThread().getId() + ".txt";
        try {
            PTK_HUIM_DFS algo = new PTK_HUIM_DFS(db, prof, tmpOut, k, true, false);
            PTK_HUIM_DFS.MiningResult result = algo.run();
            Set<String> patterns = new LinkedHashSet<>();
            for (PTK_HUIM_DFS.Pattern p : result.patterns) {
                patterns.add(patternKey(p.items));
            }
            return patterns;
        } finally {
            new File(tmpOut).delete();
        }
    }

    /**
     * Runs UTKU-PSO and returns a set of pattern keys by parsing output file.
     */
    private static Set<String> runHeuristic(String db, String prof, int k) throws Exception {
        String tmpOut = "tmp_pso_" + Thread.currentThread().getId() + ".txt";
        try {
            UTKU_PSO algo = new UTKU_PSO(db, prof, tmpOut, k, PSO_POP_SIZE, PSO_ITERATIONS);
            algo.run();
            return parsePSOPatterns(tmpOut);
        } finally {
            new File(tmpOut).delete();
        }
    }

    /**
     * Parses UTKU-PSO output file to extract pattern itemsets as string keys.
     * Output format: "Rank   {id1, id2, ...}   EU   EP"
     */
    static Set<String> parsePSOPatterns(String outputFile) throws IOException {
        Set<String> patterns = new LinkedHashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(outputFile))) {
            String line;
            boolean inResults = false;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                // Detect header line
                if (line.startsWith("Rank") && line.contains("Pattern")) {
                    inResults = true;
                    continue;
                }
                if (line.startsWith("---") || line.startsWith("===")) {
                    if (inResults && line.startsWith("===")) break;
                    continue;
                }
                if (!inResults) continue;
                if (line.isEmpty()) continue;

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
    // Output
    // =========================================================================
    private static void writeHeader(PrintWriter w) {
        w.printf("%s%n", "=".repeat(80));
        w.println("ACCURACY COMPARISON BENCHMARK (IEEE Access Section 6.C)");
        w.printf("%s%n", "=".repeat(80));
        w.printf("  Ground Truth:    PTK-HUIM-DFS (exact)%n");
        w.printf("  Heuristic:       UTKU-PSO (pop=%d, iter=%d)%n", PSO_POP_SIZE, PSO_ITERATIONS);
        w.printf("  Formula:         accuracy = |correct| / |exact| × 100%%%n");
        w.printf("  Matching:        by itemset identity (sorted item IDs)%n");
        w.printf("  Timestamp:       %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        w.printf("%s%n", "=".repeat(80));
    }

    /**
     * Writes accuracy table in IEEE Access paper format.
     */
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

        // Print header
        StringBuilder header = new StringBuilder(String.format("  %-12s", "Dataset"));
        for (int k : kList) header.append(String.format(" %8s", "k=" + k));
        w.println(header);
        w.printf("  %s%n", "-".repeat(12 + kList.size() * 9));

        // Print rows
        for (Map.Entry<String, List<AccuracyResult>> entry : byDataset.entrySet()) {
            StringBuilder row = new StringBuilder(String.format("  %-12s", entry.getKey()));
            Map<Integer, AccuracyResult> byK = new LinkedHashMap<>();
            for (AccuracyResult r : entry.getValue()) byK.put(r.k, r);

            for (int k : kList) {
                AccuracyResult r = byK.get(k);
                if (r == null) {
                    row.append(String.format(" %8s", "-"));
                } else if (r.exactFailed) {
                    row.append(String.format(" %8s", "DNF"));
                } else {
                    row.append(String.format(Locale.US, " %7.1f%%", r.accuracyPct));
                }
            }
            w.println(row);
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}