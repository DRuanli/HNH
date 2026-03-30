package experiments;

import algorithms.PTK_HUIM_BFS;
import algorithms.PTK_HUIM_BestFS;
import algorithms.PTK_HUIM_DFS;
import algorithms.UTKU_PSO;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Exp_Runtime: Runtime comparison of PTK-HUIM (DFS, BFS, BestFS) and UTKU-PSO
 * across multiple datasets and k values.
 *
 * IEEE Access Section 6.B — generates data for runtime comparison figures.
 *
 * Protocol: 2 warmup + 5 measured runs per (dataset, k, algorithm).
 * Reports mean, median, stddev, CV% for each configuration.
 * Timeout: 3600s per single run (configurable).
 *
 * Usage: java experiments.Exp_Runtime [--output-dir results/] [--timeout 3600]
 */
public class Exp_Runtime {

    // =========================================================================
    // Constants
    // =========================================================================
    private static final int WARMUP_RUNS = 2;
    private static final int MEASURED_RUNS = 5;
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

    private static final String[] ALGO_NAMES = {"PTK_DFS", "PTK_BFS", "PTK_BestFS", "UTKU_PSO"};

    // =========================================================================
    // Result container
    // =========================================================================
    static class BenchmarkResult {
        String algorithm;
        String dataset;
        int k;
        int patternCount;
        double memoryMB;
        long[] runTimes;
        double mean, median, stddev;
        long min, max;
        boolean timedOut;
        boolean error;
        String errorMsg;
    }

    // =========================================================================
    // Main
    // =========================================================================
    public static void main(String[] args) throws Exception {
        String outputDir = "results/";
        long timeoutMs = 3600_000;

        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--output-dir=")) outputDir = args[i].substring(13);
            else if (args[i].startsWith("--timeout=")) timeoutMs = Long.parseLong(args[i].substring(10)) * 1000;
        }

        new File(outputDir).mkdirs();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String outFile = outputDir + "runtime_" + timestamp + ".txt";

        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(outFile)))) {
            writeHeader(w, timeoutMs);

            for (String[] ds : DATASETS) {
                String name = ds[0], db = ds[1], prof = ds[2];
                int[] kVals = K_VALUES.get(name);
                if (kVals == null) continue;

                // Verify dataset files exist
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
                    w.printf("%n--- k = %d ---%n", k);
                    System.out.printf("  k = %d%n", k);

                    List<BenchmarkResult> results = new ArrayList<>();

                    // Run each algorithm
                    for (String algoName : ALGO_NAMES) {
                        BenchmarkResult r = runBenchmark(algoName, db, prof, k, timeoutMs);
                        r.dataset = name;
                        r.k = k;
                        results.add(r);
                    }

                    // Write results for this (dataset, k)
                    writeSummaryTable(w, results);
                    w.flush();
                }
            }

            w.printf("%n%s%n", "=".repeat(80));
            w.println("EXPERIMENT COMPLETE");
            w.printf("Timestamp: %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            w.printf("%s%n", "=".repeat(80));
        }

        System.out.printf("%nResults saved to: %s%n", outFile);
    }

    // =========================================================================
    // Benchmark Runner
    // =========================================================================
    private static BenchmarkResult runBenchmark(String algoName, String db, String prof,
                                                 int k, long timeoutMs) {
        BenchmarkResult result = new BenchmarkResult();
        result.algorithm = algoName;

        // Warmup
        System.out.printf("    [%s] Warmup: ", algoName);
        for (int i = 0; i < WARMUP_RUNS; i++) {
            System.out.printf("%d ", i + 1);
            System.out.flush();
            try {
                runSingle(algoName, db, prof, k);
            } catch (Exception e) { /* ignore warmup errors */ }
            System.gc();
            sleep(200);
        }
        System.out.println();

        // Measured runs
        System.out.printf("    [%s] Measured: ", algoName);
        long[] times = new long[MEASURED_RUNS];
        int lastCount = 0;
        double lastMem = 0;

        for (int i = 0; i < MEASURED_RUNS; i++) {
            System.out.printf("%d ", i + 1);
            System.out.flush();
            try {
                long start = System.currentTimeMillis();
                Object[] res = runSingle(algoName, db, prof, k);
                long elapsed = System.currentTimeMillis() - start;

                if (elapsed > timeoutMs) {
                    result.timedOut = true;
                    System.out.print("TIMEOUT ");
                    break;
                }

                times[i] = elapsed;
                lastCount = (int) res[0];
                lastMem = (double) res[1];
            } catch (Exception e) {
                result.error = true;
                result.errorMsg = e.getMessage();
                System.out.print("ERROR ");
                break;
            }
            System.gc();
            sleep(200);
        }
        System.out.println();

        result.runTimes = times;
        result.patternCount = lastCount;
        result.memoryMB = lastMem;

        if (!result.timedOut && !result.error) {
            computeStats(result);
        }

        return result;
    }

    /**
     * Runs a single algorithm execution, returns {patternCount, memoryMB}.
     */
    private static Object[] runSingle(String algoName, String db, String prof,
                                       int k) throws Exception {
        String tmpOut = "tmp_" + algoName + "_" + Thread.currentThread().getId() + ".txt";
        try {
            switch (algoName) {
                case "PTK_DFS": {
                    PTK_HUIM_DFS algo = new PTK_HUIM_DFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_DFS.MiningResult r = algo.run();
                    return new Object[]{r.patterns.size(), r.memoryUsedMB};
                }
                case "PTK_BFS": {
                    PTK_HUIM_BFS algo = new PTK_HUIM_BFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_BFS.MiningResult r = algo.run();
                    return new Object[]{r.patterns.size(), r.memoryUsedMB};
                }
                case "PTK_BestFS": {
                    PTK_HUIM_BestFS algo = new PTK_HUIM_BestFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_BestFS.MiningResult r = algo.run();
                    return new Object[]{r.patterns.size(), r.memoryUsedMB};
                }
                case "UTKU_PSO": {
                    UTKU_PSO algo = new UTKU_PSO(db, prof, tmpOut, k, PSO_POP_SIZE, PSO_ITERATIONS);
                    UTKU_PSO.MiningResult r = algo.run();
                    return new Object[]{r.patternCount, r.memoryUsedMB};
                }
                default:
                    throw new IllegalArgumentException("Unknown algorithm: " + algoName);
            }
        } finally {
            new File(tmpOut).delete();
        }
    }

    /**
     * Parses UTKU_PSO output file to extract pattern count and memory.
     */
    static Object[] parsePSOOutput(String outputFile) throws IOException {
        int count = 0;
        double memory = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(outputFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("Patterns found:")) {
                    count = Integer.parseInt(line.split(":")[1].trim());
                } else if (line.startsWith("Memory used:")) {
                    String val = line.split(":")[1].trim().replace("MB", "").trim();
                    memory = Double.parseDouble(val);
                }
            }
        }
        return new Object[]{count, memory};
    }

    // =========================================================================
    // Statistics
    // =========================================================================
    private static void computeStats(BenchmarkResult r) {
        long[] valid = Arrays.stream(r.runTimes).filter(t -> t > 0 || r.runTimes[0] == 0).toArray();
        if (valid.length == 0) return;

        r.mean = Arrays.stream(valid).average().orElse(0);
        long[] sorted = valid.clone();
        Arrays.sort(sorted);
        r.median = sorted[sorted.length / 2];
        r.min = sorted[0];
        r.max = sorted[sorted.length - 1];
        double variance = Arrays.stream(valid).mapToDouble(t -> Math.pow(t - r.mean, 2)).average().orElse(0);
        r.stddev = Math.sqrt(variance);
    }

    // =========================================================================
    // Output
    // =========================================================================
    private static void writeHeader(PrintWriter w, long timeoutMs) {
        w.printf("%s%n", "=".repeat(80));
        w.println("RUNTIME COMPARISON BENCHMARK (IEEE Access Section 6.B)");
        w.printf("%s%n", "=".repeat(80));
        w.printf("  JDK:             %s%n", System.getProperty("java.version"));
        w.printf("  OS:              %s %s%n", System.getProperty("os.name"), System.getProperty("os.arch"));
        w.printf("  Processors:      %d%n", Runtime.getRuntime().availableProcessors());
        w.printf("  Max Heap:        %d MB%n", Runtime.getRuntime().maxMemory() / (1024 * 1024));
        w.printf("  PSO Pop Size:    %d%n", PSO_POP_SIZE);
        w.printf("  PSO Iterations:  %d%n", PSO_ITERATIONS);
        w.printf("  Warmup Runs:     %d%n", WARMUP_RUNS);
        w.printf("  Measured Runs:   %d%n", MEASURED_RUNS);
        w.printf("  Timeout:         %d s%n", timeoutMs / 1000);
        w.printf("  Timestamp:       %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        w.printf("%s%n", "=".repeat(80));
    }

    private static void writeSummaryTable(PrintWriter w, List<BenchmarkResult> results) {
        w.printf("%n  %-15s %8s %12s %12s %10s %7s  %s%n",
                "Algorithm", "Patterns", "Mean(ms)", "Median(ms)", "StdDev", "CV%", "Runs(ms)");
        w.printf("  %s%n", "-".repeat(90));
        for (BenchmarkResult r : results) {
            if (r.timedOut) {
                w.printf("  %-15s %8s %12s%n", r.algorithm, "DNF", "TIMEOUT");
            } else if (r.error) {
                w.printf("  %-15s %8s %12s  %s%n", r.algorithm, "ERR", "ERROR", r.errorMsg);
            } else {
                double cv = r.mean > 0 ? (r.stddev / r.mean) * 100 : 0;
                String runs = Arrays.toString(r.runTimes);
                w.printf(Locale.US, "  %-15s %8d %12.1f %12.1f %10.1f %6.1f%%  %s%n",
                        r.algorithm, r.patternCount, r.mean, r.median, r.stddev, cv, runs);
            }
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}