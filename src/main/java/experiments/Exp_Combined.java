package experiments;

import algorithms.PTK_HUIM_DFS;
import algorithms.PTK_HUIM_BFS;
import algorithms.PTK_HUIM_BestFS;
import algorithms.UTKU_PSO;

import java.io.*;
import java.lang.management.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import com.sun.management.GarbageCollectionNotificationInfo;
import javax.management.*;
import javax.management.openmbean.CompositeData;

/**
 * Exp_Combined: Runs both Accuracy and Memory experiments in a single pass with
 * multiple runs for statistical validity (IEEE Access compliant).
 *
 * Protocol: 1 warmup + N measured runs per (dataset, k, algorithm).
 * GC contamination guard: if a full (major) GC fires during a measured run,
 * that run's memory snapshot is discarded and the run is repeated — up to
 * MAX_RETRY_ATTEMPTS extra tries. This prevents GC-induced low-memory artifacts
 * (e.g., the PTK-BFS k=50 Run1=79 MB anomaly) from polluting reported statistics.
 *
 * Launch flags (REQUIRED for reproducible memory numbers):
 *   java -Xms8g -Xmx8g -XX:+UseG1GC -cp target/classes experiments.Exp_Combined
 *
 * Outputs:
 *   - accuracy_<timestamp>.txt  (IEEE Access Section 6.C)
 *   - memory_<timestamp>.txt    (IEEE Access Section 6.D)
 *
 * Usage: java experiments.Exp_Combined [--output-dir=results/]
 */
public class Exp_Combined {

    // =========================================================================
    // Experimental Protocol Constants
    // =========================================================================
    private static final int WARMUP_RUNS                 = 1;
    private static final int MEASURED_RUNS_DETERMINISTIC = 3;   // PTK_DFS / BFS / BestFS
    private static final int MEASURED_RUNS_STOCHASTIC    = 5;   // UTKU_PSO
    private static final int MAX_RETRY_ATTEMPTS          = 5;   // extra retries if GC fires

    private static final int PSO_POP_SIZE  = 20;
    private static final int PSO_ITERATIONS = 10000;

    private static final String[][] DATASETS = {
        {"Chess",     "src/data/chess_database.txt",     "src/data/chess_profit.txt"},
        //{"Mushroom",  "src/data/mushroom_database.txt",  "src/data/mushroom_profit.txt"},
        //{"Accidents", "src/data/accidents_database.txt", "src/data/accidents_profit.txt"},
        //{"Retail",    "src/data/retail_database.txt",    "src/data/retail_profit.txt"},
        //{"Kosarak",   "src/data/kosarak_database.txt",   "src/data/kosarak_profit.txt"},
        //{"Pumsb",     "src/data/pumsb_database.txt",     "src/data/pumsb_profit.txt"},
    };

    private static final Map<String, int[]> K_VALUES = new LinkedHashMap<>();
    static {
        K_VALUES.put("Chess",     new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Mushroom",  new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Retail",    new int[]{10, 100, 1000, 5000, 10000, 20000});
        K_VALUES.put("Accidents", new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Kosarak",   new int[]{1, 10, 50, 100, 150, 200});
        K_VALUES.put("Pumsb",     new int[]{1, 10, 50, 100, 150, 200});
    }

    private static final String[] ALL_ALGOS = {"PTK_DFS", "PTK_BFS", "PTK_BestFS", "UTKU_PSO"};

    // =========================================================================
    // Inner class: GC contamination guard
    // =========================================================================
    /**
     * Attaches to all GarbageCollectorMXBeans and counts MAJOR (full) GC events
     * that fire while the guard is active.
     *
     * Only GCs whose cause is NOT "System.gc()" are counted — our own explicit
     * System.gc() calls between runs are intentional cleanup, not contamination.
     *
     * Usage:
     *   GcGuard guard = new GcGuard();
     *   guard.start();
     *   ... run algorithm ...
     *   guard.stop();
     *   if (guard.isContaminated()) { // discard this run }
     */
    static class GcGuard {
        private final List<NotificationEmitter>  emitters  = new ArrayList<>();
        private final List<NotificationListener> listeners = new ArrayList<>();
        private final AtomicInteger fullGcCount  = new AtomicInteger(0);
        private final AtomicInteger youngGcCount = new AtomicInteger(0);

        void start() {
            fullGcCount.set(0);
            youngGcCount.set(0);
            emitters.clear();
            listeners.clear();

            for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
                if (!(gcBean instanceof NotificationEmitter)) continue;

                NotificationEmitter emitter = (NotificationEmitter) gcBean;

                NotificationListener listener = (notif, handback) -> {
                    if (!GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION
                            .equals(notif.getType())) return;

                    GarbageCollectionNotificationInfo info =
                            GarbageCollectionNotificationInfo.from(
                                    (CompositeData) notif.getUserData());

                    String action = info.getGcAction(); // "end of major GC" or "end of minor GC"
                    String cause  = info.getGcCause();  // "System.gc()", "Allocation Failure", etc.

                    if (action.contains("major") || action.contains("Full")) {
                        // Ignore our own explicit System.gc() calls — they are between runs
                        if (!"System.gc()".equals(cause)) {
                            fullGcCount.incrementAndGet();
                        }
                    } else {
                        youngGcCount.incrementAndGet();
                    }
                };

                emitter.addNotificationListener(listener, null, null);
                emitters.add(emitter);
                listeners.add(listener);
            }
        }

        void stop() {
            for (int i = 0; i < emitters.size(); i++) {
                try { emitters.get(i).removeNotificationListener(listeners.get(i)); }
                catch (ListenerNotFoundException ignored) {}
            }
        }

        int  fullGcCount()      { return fullGcCount.get(); }
        int  youngGcCount()     { return youngGcCount.get(); }
        boolean isContaminated(){ return fullGcCount.get() > 0; }
    }

    // =========================================================================
    // Result containers
    // =========================================================================
    static class AccuracyResult {
        String  dataset;
        int     k;
        int     exactCount;
        int     heuristicCount;
        double  meanAccuracyPct;
        double  stddevAccuracy;
        double  minAccuracyPct;
        double  maxAccuracyPct;
        long    exactTimeMs;
        long    heuristicTimeMs;
        int     runs;
        double[] accuracyRuns;   // per-run accuracy values
        boolean exactFailed;
        String  failReason;
    }

    static class MemoryResult {
        String  algorithm;
        String  dataset;
        int     k;
        double  meanMemoryMB;
        double  stddevMemoryMB;
        double  peakMemoryMB;    // max across all accepted runs
        int     patternCount;
        long    meanRuntimeMs;
        int     runs;
        double[] memoryRuns;     // per-run memory (accepted runs only)
        int[]   gcCountPerRun;   // full-GC count per accepted run (should all be 0)
        int     discardedRuns;   // runs discarded due to GC contamination
        List<Set<String>> allPatterns = new ArrayList<>(); // patterns per run (for accuracy)
        boolean failed;
        String  failReason;
    }

    /** Raw result from one algorithm invocation. */
    static class AlgoRunResult {
        Set<String> patterns   = new LinkedHashSet<>();
        int         patternCount;
        double      peakMemoryMB;
        long        runtimeMs;
        int         fullGcCount  = 0;
        int         youngGcCount = 0;
        boolean     failed;
        String      failReason;
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
        String timestamp    = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String accuracyFile = outputDir + "accuracy_" + timestamp + ".txt";
        String memoryFile   = outputDir + "memory_"   + timestamp + ".txt";

        List<AccuracyResult> accuracyResults = new ArrayList<>();
        List<MemoryResult>   memoryResults   = new ArrayList<>();
        int totalCorrect = 0, totalK = 0;

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

                String dsHeader = String.format("%n%s%nDATASET: %s%n%s",
                        "=".repeat(80), name, "=".repeat(80));
                accW.println(dsHeader);
                memW.println(dsHeader);
                System.out.printf("%n=== DATASET: %s ===%n", name);

                for (int k : kVals) {
                    System.out.printf("  k = %d%n", k);
                    memW.printf("%n--- k = %d ---%n", k);

                    Map<String, MemoryResult> memResults = new LinkedHashMap<>();

                    for (String algoName : ALL_ALGOS) {
                        System.out.printf("    [%s] ", algoName);
                        System.out.flush();

                        MemoryResult mr = runAlgorithmMultiple(algoName, db, prof, k);
                        mr.dataset = name;
                        mr.k       = k;
                        memResults.put(algoName, mr);
                        memoryResults.add(mr);

                        if (mr.failed) {
                            System.out.printf("FAILED (%s)%n", mr.failReason);
                        } else {
                            System.out.printf(Locale.US,
                                    "mean=%.2f MB (±%.2f), peak=%.2f MB, discarded=%d%n",
                                    mr.meanMemoryMB, mr.stddevMemoryMB,
                                    mr.peakMemoryMB, mr.discardedRuns);
                        }

                        System.gc();
                        sleep(300);
                    }

                    // Accuracy: compare PTK_DFS ground truth vs UTKU_PSO
                    AccuracyResult ar = calculateAccuracy(name, k, memResults, accW);
                    accuracyResults.add(ar);

                    if (!ar.exactFailed) {
                        totalCorrect += (int)(ar.meanAccuracyPct * ar.exactCount / 100.0);
                        totalK       += ar.exactCount;
                    }

                    writeMemoryRow(memW, memResults);

                    System.gc();
                    sleep(300);
                }
            }

            // ── Accuracy summary ──────────────────────────────────────────────
            accW.printf("%n%n%s%n", "=".repeat(80));
            accW.println("ACCURACY SUMMARY TABLE (for paper)");
            accW.printf("%s%n%n", "=".repeat(80));
            writeAccuracyTable(accW, accuracyResults);
            double overallAccuracy = totalK > 0 ? (double) totalCorrect / totalK * 100.0 : 0.0;
            accW.printf(Locale.US, "%nOverall mean accuracy: %.1f%%%n", overallAccuracy);
            accW.printf("%n%s%n", "=".repeat(80));

            // ── Memory summary ────────────────────────────────────────────────
            memW.printf("%n%n%s%n", "=".repeat(80));
            memW.println("MEMORY SUMMARY TABLE (for paper figures)");
            memW.printf("%s%n", "=".repeat(80));
            writeMemorySummary(memW, memoryResults);
        }

        System.out.printf("%nResults saved to:%n  - %s%n  - %s%n", accuracyFile, memoryFile);
    }

    // =========================================================================
    // Multi-run execution with GC contamination filtering
    // =========================================================================

    /**
     * Runs an algorithm multiple times and computes statistics.
     * Runs where a full (major) GC fired are discarded and retried.
     */
    private static MemoryResult runAlgorithmMultiple(String algoName, String db, String prof, int k) {
        MemoryResult mr = new MemoryResult();
        mr.algorithm = algoName;

        int targetRuns = algoName.equals("UTKU_PSO")
                ? MEASURED_RUNS_STOCHASTIC
                : MEASURED_RUNS_DETERMINISTIC;

        // ── Warmup ────────────────────────────────────────────────────────────
        System.out.printf("warmup ");
        for (int i = 0; i < WARMUP_RUNS; i++) {
            try { runAlgorithmOnce(algoName, db, prof, k); System.out.printf("."); }
            catch (Exception e) { /* ignore warmup failures */ }
            System.gc();
            sleep(200);
        }

        // ── Measured runs with GC retry ───────────────────────────────────────
        System.out.printf(" measure ");
        List<AlgoRunResult> accepted = new ArrayList<>();
        List<AlgoRunResult> rejected = new ArrayList<>();
        int attempts = 0;
        int maxAttempts = targetRuns + MAX_RETRY_ATTEMPTS;

        while (accepted.size() < targetRuns && attempts < maxAttempts) {
            attempts++;
            System.out.printf(".");
            System.out.flush();

            AlgoRunResult result = runAlgorithmOnce(algoName, db, prof, k);

            if (result.failed) {
                mr.failed     = true;
                mr.failReason = result.failReason;
                return mr;
            }

            if (result.fullGcCount > 0) {
                // Major GC fired → memory snapshot unreliable → discard
                System.out.printf("[GC!×%d]", result.fullGcCount);
                rejected.add(result);
            } else {
                accepted.add(result);
            }

            System.gc();
            sleep(300);
        }

        // If we could not collect enough clean runs, fall back gracefully
        if (accepted.size() < targetRuns) {
            System.out.printf("[WARN: only %d/%d clean runs — using contaminated fallback] ",
                    accepted.size(), targetRuns);
            accepted.addAll(rejected);
        }

        System.out.printf(" ");

        // ── Compute statistics over accepted runs ─────────────────────────────
        mr.runs            = accepted.size();
        mr.memoryRuns      = new double[mr.runs];
        mr.gcCountPerRun   = new int[mr.runs];
        mr.discardedRuns   = rejected.size();
        mr.patternCount    = accepted.get(0).patternCount;

        double sumMemory = 0, sumTime = 0;
        mr.peakMemoryMB = 0;

        for (int i = 0; i < accepted.size(); i++) {
            AlgoRunResult r = accepted.get(i);
            mr.memoryRuns[i]    = r.peakMemoryMB;
            mr.gcCountPerRun[i] = r.fullGcCount;
            sumMemory          += r.peakMemoryMB;
            sumTime            += r.runtimeMs;
            mr.peakMemoryMB     = Math.max(mr.peakMemoryMB, r.peakMemoryMB);

            if (!r.patterns.isEmpty()) {
                mr.allPatterns.add(r.patterns);
            }
        }

        mr.meanMemoryMB  = sumMemory / mr.runs;
        mr.meanRuntimeMs = (long)(sumTime / mr.runs);

        double variance = 0;
        for (double mem : mr.memoryRuns) {
            variance += Math.pow(mem - mr.meanMemoryMB, 2);
        }
        mr.stddevMemoryMB = Math.sqrt(variance / mr.runs);

        return mr;
    }

    /**
     * Runs an algorithm exactly once.
     * Wraps execution with the GcGuard to detect full GC events.
     */
    private static AlgoRunResult runAlgorithmOnce(String algoName, String db, String prof, int k) {
        AlgoRunResult result = new AlgoRunResult();

        // Force GC before measurement.
        // Using System.gc() here triggers a "System.gc()" cause, which the GcGuard ignores.
        System.gc();
        sleep(100);

        GcGuard guard  = new GcGuard();
        String  tmpOut = "tmp_combined_" + algoName + "_" + System.nanoTime() + ".txt";

        guard.start();
        try {
            long start = System.currentTimeMillis();

            switch (algoName) {
                case "PTK_DFS": {
                    PTK_HUIM_DFS algo = new PTK_HUIM_DFS(db, prof, tmpOut, k, true, false);
                    PTK_HUIM_DFS.MiningResult r = algo.run();
                    result.peakMemoryMB = r.memoryUsedMB;
                    result.patternCount = r.patterns.size();
                    for (PTK_HUIM_DFS.Pattern p : r.patterns)
                        result.patterns.add(patternKey(p.items));
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
                    result.patterns     = parsePSOPatterns(tmpOut);
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unknown algorithm: " + algoName);
            }

            result.runtimeMs = System.currentTimeMillis() - start;

        } catch (Exception e) {
            result.failed     = true;
            result.failReason = e.getClass().getSimpleName() + ": " + e.getMessage();
        } finally {
            guard.stop();
            result.fullGcCount  = guard.fullGcCount();
            result.youngGcCount = guard.youngGcCount();
            new File(tmpOut).delete();
        }

        return result;
    }

    // =========================================================================
    // Accuracy calculation
    // =========================================================================

    /**
     * Compares UTKU_PSO patterns (multiple stochastic runs) against
     * PTK_DFS ground truth (deterministic — first run used as reference).
     */
    private static AccuracyResult calculateAccuracy(String dataset, int k,
                                                     Map<String, MemoryResult> memResults,
                                                     PrintWriter accW) {
        AccuracyResult ar = new AccuracyResult();
        ar.dataset = dataset;
        ar.k       = k;

        MemoryResult exactMem     = memResults.get("PTK_DFS");
        MemoryResult heuristicMem = memResults.get("UTKU_PSO");

        if (exactMem.failed) {
            ar.exactFailed = true;
            ar.failReason  = exactMem.failReason;
            accW.printf("  k=%-6d  EXACT FAILED: %s%n", k, exactMem.failReason);
            return ar;
        }
        if (heuristicMem.failed) {
            ar.exactFailed = true;
            ar.failReason  = "HEURISTIC: " + heuristicMem.failReason;
            accW.printf("  k=%-6d  HEURISTIC FAILED: %s%n", k, heuristicMem.failReason);
            return ar;
        }
        if (exactMem.allPatterns.isEmpty() || heuristicMem.allPatterns.isEmpty()) {
            ar.exactFailed = true;
            ar.failReason  = "No patterns available for comparison";
            accW.printf("  k=%-6d  ERROR: No patterns available%n", k);
            return ar;
        }

        // PTK_DFS is deterministic — use first accepted run as ground truth
        Set<String> groundTruth = exactMem.allPatterns.get(0);
        int effectiveK          = Math.max(groundTruth.size(), 1);

        ar.exactCount      = effectiveK;
        ar.heuristicCount  = heuristicMem.patternCount;
        ar.exactTimeMs     = heuristicMem.meanRuntimeMs;
        ar.heuristicTimeMs = heuristicMem.meanRuntimeMs;
        ar.runs            = heuristicMem.allPatterns.size();
        ar.accuracyRuns    = new double[ar.runs];

        double sumAcc = 0, minAcc = 100.0, maxAcc = 0.0;

        for (int i = 0; i < heuristicMem.allPatterns.size(); i++) {
            Set<String> psoPatterns = heuristicMem.allPatterns.get(i);
            int correct = 0;
            for (String p : psoPatterns) {
                if (groundTruth.contains(p)) correct++;
            }
            double acc          = (double) correct / effectiveK * 100.0;
            ar.accuracyRuns[i]  = acc;
            sumAcc             += acc;
            minAcc              = Math.min(minAcc, acc);
            maxAcc              = Math.max(maxAcc, acc);
        }

        ar.meanAccuracyPct = sumAcc / ar.runs;
        ar.minAccuracyPct  = minAcc;
        ar.maxAccuracyPct  = maxAcc;

        double variance = 0;
        for (double acc : ar.accuracyRuns)
            variance += Math.pow(acc - ar.meanAccuracyPct, 2);
        ar.stddevAccuracy = Math.sqrt(variance / ar.runs);

        // ── Write to accuracy file ────────────────────────────────────────────
        accW.printf("  k=%-6d  exact=%d  heuristic=%d%n", k, ar.exactCount, ar.heuristicCount);
        accW.printf(Locale.US,
                "           mean accuracy=%.1f%% (±%.1f%%)  range=[%.1f%%, %.1f%%]%n",
                ar.meanAccuracyPct, ar.stddevAccuracy, ar.minAccuracyPct, ar.maxAccuracyPct);
        accW.printf("           runs: ");
        for (int i = 0; i < ar.accuracyRuns.length; i++) {
            accW.printf(Locale.US, "%.1f%%", ar.accuracyRuns[i]);
            if (i < ar.accuracyRuns.length - 1) accW.printf(", ");
        }
        accW.printf("%n");

        return ar;
    }

    // =========================================================================
    // Pattern utilities
    // =========================================================================

    /**
     * Parses UTKU-PSO output file and returns pattern keys for accuracy comparison.
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
                if (line.startsWith("===") && inResults) break;
                if (!inResults || line.isEmpty() || line.startsWith("---")) continue;

                int braceStart = line.indexOf('{');
                int braceEnd   = line.indexOf('}');
                if (braceStart >= 0 && braceEnd > braceStart) {
                    List<Integer> items = new ArrayList<>();
                    for (String s : line.substring(braceStart + 1, braceEnd).split(",")) {
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
        w.printf("  GC Guard:        Runs with major GC events are discarded and retried%n");
        w.printf("  Warmup Runs:     %d%n", WARMUP_RUNS);
        w.printf("  Measured Runs:   %d (deterministic), %d (stochastic)%n",
                MEASURED_RUNS_DETERMINISTIC, MEASURED_RUNS_STOCHASTIC);
        w.printf("  Max Retries:     %d extra attempts if GC fires%n", MAX_RETRY_ATTEMPTS);
        w.printf("  JDK:             %s%n", System.getProperty("java.version"));
        w.printf("  Max Heap:        %d MB%n", Runtime.getRuntime().maxMemory() / (1024 * 1024));
        w.printf("  Timestamp:       %s%n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        w.printf("%s%n", "=".repeat(80));
    }

    private static void writeMemoryRow(PrintWriter w, Map<String, MemoryResult> results) {
        w.printf("  %-15s %8s %15s %15s %10s  %-40s  %s%n",
                "Algorithm", "Patterns", "Mean(MB)", "StdDev(MB)", "Peak(MB)",
                "Individual Runs (MB)", "GC-Discarded");
        w.printf("  %s%n", "-".repeat(130));

        for (String algo : ALL_ALGOS) {
            MemoryResult r = results.get(algo);
            if (r == null) continue;
            if (r.failed) {
                w.printf("  %-15s %8s %15s  %s%n", algo, "ERR", "FAILED", r.failReason);
                continue;
            }

            StringBuilder runStr = new StringBuilder("[");
            for (int i = 0; i < r.memoryRuns.length; i++) {
                runStr.append(String.format(Locale.US, "%.2f", r.memoryRuns[i]));
                if (i < r.memoryRuns.length - 1) runStr.append(", ");
            }
            runStr.append("]");

            w.printf(Locale.US, "  %-15s %8d %15.2f %15.2f %10.2f  %-40s  %d runs discarded%n",
                    algo, r.patternCount, r.meanMemoryMB, r.stddevMemoryMB, r.peakMemoryMB,
                    runStr, r.discardedRuns);
        }
    }

    private static void writeAccuracyTable(PrintWriter w, List<AccuracyResult> results) {
        // Group by dataset
        Map<String, List<AccuracyResult>> byDataset = new LinkedHashMap<>();
        for (AccuracyResult r : results)
            byDataset.computeIfAbsent(r.dataset, x -> new ArrayList<>()).add(r);

        Set<Integer> allKs = new LinkedHashSet<>();
        for (AccuracyResult r : results) allKs.add(r.k);
        List<Integer> kList = new ArrayList<>(allKs);

        // Summary table header
        w.println("  Summary Table (Mean Accuracy %):");
        StringBuilder header = new StringBuilder(String.format("  %-20s", "Dataset"));
        for (int k : kList) header.append(String.format(" %12s", "k=" + k));
        w.println(header);
        w.printf("  %s%n", "-".repeat(20 + kList.size() * 13));

        for (Map.Entry<String, List<AccuracyResult>> entry : byDataset.entrySet()) {
            StringBuilder row = new StringBuilder(String.format("  %-20s", entry.getKey()));
            Map<Integer, AccuracyResult> byK = new LinkedHashMap<>();
            for (AccuracyResult r : entry.getValue()) byK.put(r.k, r);
            for (int k : kList) {
                AccuracyResult r = byK.get(k);
                if (r == null)           row.append(String.format(" %12s", "-"));
                else if (r.exactFailed)  row.append(String.format(" %12s", "DNF"));
                else                     row.append(String.format(Locale.US, " %11.1f%%", r.meanAccuracyPct));
            }
            w.println(row);
        }

        // Detailed run-by-run table
        w.printf("%n%n  Detailed Run-by-Run Accuracy Data:%n");
        w.printf("  %s%n", "=".repeat(78));

        for (Map.Entry<String, List<AccuracyResult>> entry : byDataset.entrySet()) {
            w.printf("%n  Dataset: %s%n", entry.getKey());
            for (AccuracyResult r : entry.getValue()) {
                if (r.exactFailed) {
                    w.printf("    k=%-6d: FAILED - %s%n", r.k, r.failReason);
                } else if (r.accuracyRuns != null) {
                    w.printf("    k=%-6d: ", r.k);
                    for (int i = 0; i < r.accuracyRuns.length; i++) {
                        w.printf(Locale.US, "Run%d=%.1f%%", i + 1, r.accuracyRuns[i]);
                        if (i < r.accuracyRuns.length - 1) w.printf(", ");
                    }
                    w.printf(Locale.US, "  | Mean=%.1f%%, StdDev=%.1f%%, Range=[%.1f%%-%.1f%%]%n",
                            r.meanAccuracyPct, r.stddevAccuracy,
                            r.minAccuracyPct, r.maxAccuracyPct);
                }
            }
        }
    }

    private static void writeMemorySummary(PrintWriter w, List<MemoryResult> results) {
        // Group: dataset → k → algo → MemoryResult
        Map<String, Map<Integer, Map<String, MemoryResult>>> grouped = new LinkedHashMap<>();
        for (MemoryResult r : results) {
            grouped.computeIfAbsent(r.dataset, x -> new LinkedHashMap<>())
                   .computeIfAbsent(r.k,       x -> new LinkedHashMap<>())
                   .put(r.algorithm, r);
        }

        for (Map.Entry<String, Map<Integer, Map<String, MemoryResult>>> dsEntry : grouped.entrySet()) {
            w.printf("%n  Dataset: %s (Mean Memory ± StdDev MB)%n", dsEntry.getKey());
            w.printf("  %-8s", "k");
            for (String algo : ALL_ALGOS) w.printf(" %20s", algo);
            w.println();
            w.printf("  %s%n", "-".repeat(8 + ALL_ALGOS.length * 21));

            for (Map.Entry<Integer, Map<String, MemoryResult>> kEntry : dsEntry.getValue().entrySet()) {
                w.printf("  %-8d", kEntry.getKey());
                for (String algo : ALL_ALGOS) {
                    MemoryResult r = kEntry.getValue().get(algo);
                    if (r == null || r.failed) {
                        w.printf(" %20s", "DNF");
                    } else {
                        w.printf(Locale.US, " %13.2f (±%5.2f)", r.meanMemoryMB, r.stddevMemoryMB);
                    }
                }
                w.println();
            }
        }

        // Detailed run-by-run section
        w.printf("%n%n%s%n", "=".repeat(80));
        w.println("DETAILED RUN-BY-RUN MEMORY DATA");
        w.printf("%s%n", "=".repeat(80));

        for (Map.Entry<String, Map<Integer, Map<String, MemoryResult>>> dsEntry : grouped.entrySet()) {
            w.printf("%n  Dataset: %s%n", dsEntry.getKey());

            for (Map.Entry<Integer, Map<String, MemoryResult>> kEntry : dsEntry.getValue().entrySet()) {
                w.printf("%n    k = %d:%n", kEntry.getKey());

                for (String algo : ALL_ALGOS) {
                    MemoryResult r = kEntry.getValue().get(algo);
                    if (r == null || r.failed) continue;

                    w.printf("      %-14s: ", algo);
                    for (int i = 0; i < r.memoryRuns.length; i++) {
                        w.printf(Locale.US, "Run%d=%.2f MB", i + 1, r.memoryRuns[i]);
                        if (r.gcCountPerRun != null && r.gcCountPerRun[i] > 0)
                            w.printf("[GC!×%d]", r.gcCountPerRun[i]);
                        if (i < r.memoryRuns.length - 1) w.printf(", ");
                    }
                    w.printf(Locale.US,
                            "  | Mean=%.2f, StdDev=%.2f, Peak=%.2f, Discarded=%d%n",
                            r.meanMemoryMB, r.stddevMemoryMB, r.peakMemoryMB, r.discardedRuns);
                }
            }
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}