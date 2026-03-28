package experiments;

import algorithms.PTK_HUIM_BFS;
import algorithms.PTK_HUIM_BestFS;
import algorithms.PTK_HUIM_DFS;
import algorithms.UTKU_PSO;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Exp_Memory: Peak memory usage comparison of all algorithms.
 *
 * IEEE Access Section 6.D — generates data for memory usage figures.
 *
 * Protocol: For each (dataset, k, algorithm):
 *   - Force GC before run
 *   - Run algorithm, capture internal peak memory from MiningResult
 *   - For UTKU-PSO, parse memory from output file
 *   - Report peak heap usage in MB
 *
 * Memory measured using Runtime.totalMemory() - Runtime.freeMemory(),
 * sampled after each major phase (consistent with TKU-PSO paper methodology).
 *
 * Usage: java algorithms.Exp_Memory [--output-dir results/] [--min-prob 0.1]
 */
public class Exp_Memory {

    private static final double DEFAULT_MIN_PROB = 0.1;
    private static final int PSO_POP_SIZE = 20;
    private static final int PSO_ITERATIONS = 10000;

    private static final String[][] DATASETS = {
        {"Chess",    "src/data/chess_database.txt",    "src/data/chess_profit.txt"},
        {"Mushroom", "src/data/mushroom_database.txt", "src/data/mushroom_profit.txt"},
        {"Connect",  "src/data/connect_database.txt",  "src/data/connect_profit.txt"},
        {"Retail",   "src/data/retail_database.txt",   "src/data/retail_profit.txt"},
        {"Kosarak",  "src/data/kosarak_database.txt",  "src/data/kosarak_profit.txt"},
    };

    private static final Map<String, int[]> K_VALUES = new LinkedHashMap<>();
    static {
        K_VALUES.put("Chess",    new int[]{1, 10, 100, 500, 1000, 2000});
        K_VALUES.put("Mushroom", new int[]{1, 10, 100, 500, 1000, 2000});
        K_VALUES.put("Connect",  new int[]{1, 10, 100, 500, 1000, 2000});
        K_VALUES.put("Retail",   new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Kosarak",  new int[]{1, 10, 50, 100, 150, 200});
    }

    private static final String[] ALGO_NAMES = {"PTK_DFS", "PTK_BFS", "PTK_BestFS", "UTKU_PSO"};

    static class MemoryResult {
        String algorithm;
        String dataset;
        int k;
        double peakMemoryMB;
        int patternCount;
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
        String outFile = outputDir + "memory_" + timestamp + ".txt";

        List<MemoryResult> allResults = new ArrayList<>();

        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(outFile)))) {
            writeHeader(w);

            for (String[] ds : DATASETS) {
                String name = ds[0], db = ds[1], prof = ds[2];
                int[] kVals = K_VALUES.get(name);
                if (kVals == null) continue;

                if (!new File(db).exists() || !new File(prof).exists()) {
                    w.printf("%n>>> SKIPPING %s: data files not found%n", name);
                    continue;
                }

                w.printf("%n%s%n", "=".repeat(80));
                w.printf("DATASET: %s%n", name);
                w.printf("%s%n", "=".repeat(80));
                System.out.printf("%n=== DATASET: %s ===%n", name);

                for (int k : kVals) {
                    w.printf("%n--- k = %d ---%n", k);
                    System.out.printf("  k = %d%n", k);

                    List<MemoryResult> kResults = new ArrayList<>();

                    for (String algoName : ALGO_NAMES) {
                        System.out.printf("    [%s] ... ", algoName);
                        System.out.flush();

                        MemoryResult mr = measureMemory(algoName, db, prof, k);
                        mr.dataset = name;
                        mr.k = k;
                        kResults.add(mr);
                        allResults.add(mr);

                        if (mr.failed) {
                            System.out.printf("FAILED (%s)%n", mr.failReason);
                        } else {
                            System.out.printf(Locale.US, "%.2f MB (%d patterns, %d ms)%n",
                                    mr.peakMemoryMB, mr.patternCount, mr.runtimeMs);
                        }

                        // Aggressive GC between algorithms for cleaner measurement
                        System.gc();
                        sleep(500);
                    }

                    writeMemoryRow(w, kResults);
                }
            }

            // Summary table for paper
            w.printf("%n%n%s%n", "=".repeat(80));
            w.println("MEMORY SUMMARY TABLE (for paper figures)");
            w.printf("%s%n", "=".repeat(80));
            writeMemorySummary(w, allResults);
        }

        System.out.printf("%nResults saved to: %s%n", outFile);
    }

    // =========================================================================
    // Memory Measurement
    // =========================================================================
    private static MemoryResult measureMemory(String algoName, String db, String prof,
                                               int k) {
        MemoryResult mr = new MemoryResult();
        mr.algorithm = algoName;

        // Force GC before measurement
        System.gc();
        sleep(300);

        String tmpOut = "tmp_mem_" + algoName + "_" + System.nanoTime() + ".txt";
        try {
            long start = System.currentTimeMillis();

            switch (algoName) {
                case "PTK_DFS": {
                    PTK_HUIM_DFS algo = new PTK_HUIM_DFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_DFS.MiningResult r = algo.run();
                    mr.peakMemoryMB = r.memoryUsedMB;
                    mr.patternCount = r.patterns.size();
                    break;
                }
                case "PTK_BFS": {
                    PTK_HUIM_BFS algo = new PTK_HUIM_BFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_BFS.MiningResult r = algo.run();
                    mr.peakMemoryMB = r.memoryUsedMB;
                    mr.patternCount = r.patterns.size();
                    break;
                }
                case "PTK_BestFS": {
                    PTK_HUIM_BestFS algo = new PTK_HUIM_BestFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_BestFS.MiningResult r = algo.run();
                    mr.peakMemoryMB = r.memoryUsedMB;
                    mr.patternCount = r.patterns.size();
                    break;
                }
                case "UTKU_PSO": {
                    UTKU_PSO algo = new UTKU_PSO(db, prof, tmpOut, k, PSO_POP_SIZE, PSO_ITERATIONS);
                    algo.run();
                    Object[] parsed = parsePSOOutput(tmpOut);
                    mr.patternCount = (int) parsed[0];
                    mr.peakMemoryMB = (double) parsed[1];
                    break;
                }
            }

            mr.runtimeMs = System.currentTimeMillis() - start;
        } catch (Exception e) {
            mr.failed = true;
            mr.failReason = e.getClass().getSimpleName() + ": " + e.getMessage();
        } finally {
            new File(tmpOut).delete();
        }

        return mr;
    }

    static Object[] parsePSOOutput(String outputFile) throws IOException {
        int count = 0;
        double memory = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(outputFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("Patterns found:"))
                    count = Integer.parseInt(line.split(":")[1].trim());
                else if (line.startsWith("Memory used:"))
                    memory = Double.parseDouble(line.split(":")[1].trim().replace("MB", "").trim());
            }
        }
        return new Object[]{count, memory};
    }

    // =========================================================================
    // Output
    // =========================================================================
    private static void writeHeader(PrintWriter w) {
        w.printf("%s%n", "=".repeat(80));
        w.println("MEMORY USAGE BENCHMARK (IEEE Access Section 6.D)");
        w.printf("%s%n", "=".repeat(80));
        w.printf("  Measurement:     Runtime.totalMemory() - Runtime.freeMemory()%n");
        w.printf("  Sampling:        After each major phase (peak captured)%n");
        w.printf("  JDK:             %s%n", System.getProperty("java.version"));
        w.printf("  Max Heap:        %d MB%n", Runtime.getRuntime().maxMemory() / (1024 * 1024));
        w.printf("  Timestamp:       %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        w.printf("%s%n", "=".repeat(80));
    }

    private static void writeMemoryRow(PrintWriter w, List<MemoryResult> results) {
        w.printf("  %-15s %8s %12s %10s%n", "Algorithm", "Patterns", "Memory(MB)", "Time(ms)");
        w.printf("  %s%n", "-".repeat(50));
        for (MemoryResult r : results) {
            if (r.failed) {
                w.printf("  %-15s %8s %12s  %s%n", r.algorithm, "ERR", "FAILED", r.failReason);
            } else {
                w.printf(Locale.US, "  %-15s %8d %12.2f %10d%n",
                        r.algorithm, r.patternCount, r.peakMemoryMB, r.runtimeMs);
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

        for (Map.Entry<String, Map<Integer, Map<String, MemoryResult>>> dsEntry : grouped.entrySet()) {
            w.printf("%n  Dataset: %s%n", dsEntry.getKey());
            w.printf("  %-8s", "k");
            for (String algo : ALGO_NAMES) w.printf(" %12s", algo);
            w.println();
            w.printf("  %s%n", "-".repeat(8 + ALGO_NAMES.length * 13));

            for (Map.Entry<Integer, Map<String, MemoryResult>> kEntry : dsEntry.getValue().entrySet()) {
                w.printf("  %-8d", kEntry.getKey());
                for (String algo : ALGO_NAMES) {
                    MemoryResult r = kEntry.getValue().get(algo);
                    if (r == null || r.failed) {
                        w.printf(" %12s", "DNF");
                    } else {
                        w.printf(Locale.US, " %12.2f", r.peakMemoryMB);
                    }
                }
                w.println();
            }
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}