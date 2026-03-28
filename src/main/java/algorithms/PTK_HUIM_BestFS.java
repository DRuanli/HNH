package algorithms;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * PTK-HUIM (BestFS): Exact Top-K High-Utility Itemset Mining on Uncertain Databases
 *
 * A standalone, self-contained implementation for mining the K itemsets with
 * highest Expected Utility (EU) from an uncertain transaction database where
 * items have positive or negative profits and probabilistic occurrence.
 *
 * <h3>Algorithm Configuration</h3>
 * <ul>
 *   <li><b>Search strategy:</b>  Best-First Search (BestFS) prefix-growth using a
 *       max-priority queue ordered by Positive Upper Bound (PUB) descending.
 *       By always expanding the most promising node first, BestFS raises the
 *       dynamic admission threshold as fast as possible, enabling aggressive
 *       pruning of low-potential nodes still waiting in the queue.</li>
 *   <li><b>Join strategy:</b>    Two-pointer merge with inline EU/PUB aggregation
 *       and deferred EP computation (zero transcendental calls in join loop)</li>
 *   <li><b>Top-K collector:</b>  Baseline (TreeSet min-heap + HashMap dedup)</li>
 *   <li><b>Parallelism:</b>      ForkJoin work-stealing (Phase 1a, 1d, 3)</li>
 * </ul>
 *
 * <h3>Three-Phase Pipeline</h3>
 * <ol>
 *   <li><b>Phase 1 - Preprocessing:</b> Compute PTWU/EP, filter items, rank by PTWU,
 *       build single-item UPU-Lists</li>
 *   <li><b>Phase 2 - Initialization:</b> Evaluate 1-itemsets, seed the Top-K collector</li>
 *   <li><b>Phase 3 - Mining:</b> Best-First prefix-growth with three-tier pruning:
 *       PTWU (monotone UB), PUB (tighter UB), EP (anti-monotone, deferred).
 *       Stale nodes are pruned at dequeue time against the dynamic threshold.</li>
 * </ol>
 *
 * <h3>Key Difference from DFS</h3>
 * <p>DFS explores one branch completely before backtracking. BestFS maintains a
 * global priority queue and always expands the node with the highest PUB next,
 * regardless of depth. This greedy approach finds high-EU patterns early,
 * raising the threshold faster and pruning more of the search space — at the
 * cost of O(frontier) memory instead of O(depth).</p>
 *
 * <h3>Input Format</h3>
 * <pre>
 *   Database: "itemId:quantity:probability itemId:quantity:probability ..." per line
 *   Profits:  "itemId profit" per line (negative profits allowed)
 * </pre>
 *
 * @author Elio (flattened from PTK-HUIM clean architecture)
 */
public class PTK_HUIM_BestFS {

    // =========================================================================
    // Constants
    // =========================================================================

    /** Floating-point comparison tolerance for threshold checks. */
    private static final double EPSILON = 1e-10;

    /** Log-space floor to prevent denormalized underflow. */
    private static final double LOG_ZERO = -700.0;

    /** Pre-computed log(1 - epsilon) for EP accumulation fast-path. */
    private static final double LOG_ONE_MINUS_EPS = Math.log(1.0 - EPSILON);

    /** Max transactions per ForkJoin leaf task (Phase 1a, 1d). */
    private static final int LEAF_SIZE = 256;

    /** Prefix count threshold for fine-grain ForkJoin decomposition (Phase 3). */
    private static final int FINE_GRAIN_THRESHOLD = 32;

    // =========================================================================
    // Inner Classes - Data Models
    // =========================================================================

    /**
     * Represents a single uncertain transaction.
     * Each item has a quantity and an occurrence probability in (0, 1].
     */
    private static final class Transaction {
        final int tid;
        final Map<Integer, Integer> quantities;
        final Map<Integer, Double> probabilities;

        Transaction(int tid, Map<Integer, Integer> quantities, Map<Integer, Double> probabilities) {
            this.tid = tid;
            this.quantities = quantities;
            this.probabilities = probabilities;
        }

        Set<Integer> getItems() { return quantities.keySet(); }
        int getQuantity(int item) { return quantities.getOrDefault(item, 0); }
        double getProbability(int item) { return probabilities.getOrDefault(item, 0.0); }
        int getItemCount() { return quantities.size(); }

        @Override
        public String toString() {
            return String.format("Transaction{tid=%d, items=%d}", tid, getItemCount());
        }
    }

    /**
     * Per-transaction item descriptor used during UPU-List construction (Phase 1d).
     * Transient: created and consumed within a single transaction processing loop.
     */
    private static final class ItemInfo implements Comparable<ItemInfo> {
        final int itemId;
        final int rank;
        final double profit;
        final double utility;
        final double logProbability;

        ItemInfo(int itemId, int rank, double profit, int quantity, double logProb) {
            this.itemId = itemId;
            this.rank = rank;
            this.profit = profit;
            this.utility = profit * quantity;
            this.logProbability = logProb;
        }

        @Override
        public int compareTo(ItemInfo o) {
            return Integer.compare(this.rank, o.rank);
        }

        @Override
        public String toString() {
            return String.format("ItemInfo{id=%d, rank=%d, util=%.2f}", itemId, rank, utility);
        }
    }

    /**
     * Per-transaction entry collected during Phase 1d-a.
     * Represents one item's contribution in one transaction.
     */
    private static final class TransactionEntry implements Comparable<TransactionEntry> {
        final int tid;
        final double utility;
        final double remainingUtility;
        final double logProbability;

        TransactionEntry(int tid, double utility, double remaining, double logProb) {
            this.tid = tid;
            this.utility = utility;
            this.remainingUtility = remaining;
            this.logProbability = logProb;
        }

        @Override
        public int compareTo(TransactionEntry o) {
            return Integer.compare(this.tid, o.tid);
        }
    }

    /**
     * UPU-List: transactional projection of one itemset onto the database.
     *
     * Arrays are TID-sorted for O(n) two-pointer join. Pre-aggregated statistics
     * (EU, EP, PTWU, PUB) are computed during construction or deferred.
     *
     * <p>Probabilities are stored directly (not in log-space) to enable O(1)
     * multiplication during joins instead of expensive Math.exp() calls.
     * EP computation is deferred until after PTWU/PUB pruning passes,
     * avoiding costly Math.log1p() calls for joins that will be pruned.</p>
     *
     * Immutable after EP resolution - safe for concurrent reads in Phase 3.
     */
    private static final class UPUList {
        final Set<Integer> itemset;
        final int[] transactionIds;
        final double[] utilities;
        final double[] remainingUtilities;
        final double[] probabilities;
        final int entryCount;
        final double ptwu;
        final double expectedUtility;
        /** EP: set at construction for 1-itemsets, deferred for joined lists. */
        double existentialProbability;
        final double positiveUpperBound;

        UPUList(Set<Integer> itemset, int[] tids, double[] utils, double[] remaining,
                double[] probs, int count, double ptwu, double eu, double ep, double pub) {
            this.itemset = itemset;
            this.transactionIds = tids;
            this.utilities = utils;
            this.remainingUtilities = remaining;
            this.probabilities = probs;
            this.entryCount = count;
            this.ptwu = ptwu;
            this.expectedUtility = eu;
            this.existentialProbability = ep;
            this.positiveUpperBound = pub;
        }

        @Override
        public String toString() {
            return String.format("UPUList{items=%s, entries=%d, EU=%.4f, EP=%.6f}",
                    itemset, entryCount, expectedUtility, existentialProbability);
        }
    }

    // =========================================================================
    // Inner Classes - Result Models
    // =========================================================================

    /**
     * Represents a node in the BestFS priority queue frontier.
     *
     * Each node holds the UPU-List of the current prefix and the first
     * extension index to consider. Nodes are ordered by PUB descending
     * in the priority queue so the most promising prefix is expanded first.
     */
    private static final class SearchNode {
        final UPUList list;
        final int startIndex;

        SearchNode(UPUList list, int startIndex) {
            this.list = list;
            this.startIndex = startIndex;
        }
    }

    // =========================================================================
    // Inner Classes - Result Models (continued)
    // =========================================================================

    /**
     * Discovered high-utility pattern.
     * Ordered by EU ascending (min-heap). Equality by itemset identity only.
     */
    static final class Pattern implements Comparable<Pattern> {
        final Set<Integer> items;
        final double expectedUtility;
        final double existentialProbability;
        private final List<Integer> sortedItems;

        Pattern(Set<Integer> items, double eu, double ep) {
            this.items = items;
            this.expectedUtility = eu;
            this.existentialProbability = ep;
            this.sortedItems = new ArrayList<>(items);
            Collections.sort(this.sortedItems);
        }

        @Override
        public int compareTo(Pattern o) {
            int c = Double.compare(this.expectedUtility, o.expectedUtility);
            if (c != 0) return c;
            c = Integer.compare(this.items.size(), o.items.size());
            if (c != 0) return c;
            for (int i = 0; i < this.sortedItems.size(); i++) {
                c = Integer.compare(this.sortedItems.get(i), o.sortedItems.get(i));
                if (c != 0) return c;
            }
            return 0;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof Pattern)) return false;
            return items.equals(((Pattern) obj).items);
        }

        @Override
        public int hashCode() { return items.hashCode(); }

        @Override
        public String toString() {
            return String.format("Pattern{items=%s, EU=%.4f, EP=%.6f}",
                    sortedItems, expectedUtility, existentialProbability);
        }
    }

    /**
     * Encapsulates the complete mining result for clean return from run().
     */
    static final class MiningResult {
        final List<Pattern> patterns;
        final long executionTimeMs;
        final double memoryUsedMB;
        final int validItemCount;
        final int upuListCount;
        final int databaseSize;
        final long phase1Ms, phase2Ms, phase3Ms;

        MiningResult(List<Pattern> patterns, long execMs, double memMB,
                     int validItems, int upuLists, int dbSize,
                     long p1, long p2, long p3) {
            this.patterns = patterns;
            this.executionTimeMs = execMs;
            this.memoryUsedMB = memMB;
            this.validItemCount = validItems;
            this.upuListCount = upuLists;
            this.databaseSize = dbSize;
            this.phase1Ms = p1;
            this.phase2Ms = p2;
            this.phase3Ms = p3;
        }
    }

    // =========================================================================
    // Inner Classes - Top-K Collector
    // =========================================================================

    /**
     * Thread-safe Top-K pattern collector: TreeSet (EU ordering) + HashMap (dedup).
     *
     * Volatile threshold enables lock-free fast-path rejection.
     * ReentrantLock guards all mutations for TreeSet/HashMap consistency.
     */
    private static final class TopKCollector {
        private final int capacity;
        private final TreeSet<Pattern> heap = new TreeSet<>();
        private final Map<Set<Integer>, Pattern> index = new HashMap<>();
        private final ReentrantLock lock = new ReentrantLock();
        volatile double admissionThreshold = 0.0;

        TopKCollector(int k) { this.capacity = k; }

        boolean tryCollect(UPUList candidate) {
            double eu = candidate.expectedUtility;
            if (heap.size() >= capacity && eu < admissionThreshold - EPSILON) return false;

            lock.lock();
            try {
                if (heap.size() >= capacity && eu < admissionThreshold - EPSILON) return false;

                Set<Integer> itemset = candidate.itemset;
                Pattern existing = index.get(itemset);

                if (existing != null) {
                    if (eu > existing.expectedUtility + EPSILON) {
                        heap.remove(existing);
                        Pattern updated = new Pattern(itemset, eu, candidate.existentialProbability);
                        heap.add(updated);
                        index.put(itemset, updated);
                        updateThreshold();
                        return true;
                    }
                    return false;
                }

                Pattern newPattern = new Pattern(itemset, eu, candidate.existentialProbability);
                heap.add(newPattern);
                index.put(itemset, newPattern);

                while (heap.size() > capacity) {
                    Pattern weakest = heap.pollFirst();
                    index.remove(weakest.items);
                }
                updateThreshold();
                return true;
            } finally {
                lock.unlock();
            }
        }

        private void updateThreshold() {
            admissionThreshold = (heap.size() >= capacity && !heap.isEmpty())
                    ? heap.first().expectedUtility : 0.0;
        }

        List<Pattern> getResults() {
            lock.lock();
            try { return new ArrayList<>(heap.descendingSet()); }
            finally { lock.unlock(); }
        }

        double getThreshold() { return admissionThreshold; }

        int size() {
            lock.lock();
            try { return heap.size(); }
            finally { lock.unlock(); }
        }
    }

    // =========================================================================
    // Inner Classes - ForkJoin Tasks
    // =========================================================================

    /** Phase 1a: parallel PTWU + EP computation via ForkJoin. */
    private final class Phase1Task extends RecursiveTask<double[][]> {
        final int from, to;
        Phase1Task(int from, int to) { this.from = from; this.to = to; }

        @Override
        protected double[][] compute() {
            if (to - from <= LEAF_SIZE) {
                double[] ptwu = new double[denseSize];
                double[] lc = new double[denseSize];
                for (int i = from; i < to; i++)
                    processTransactionPhase1(database.get(i), ptwu, lc);
                return new double[][] { ptwu, lc };
            }
            int mid = (from + to) >>> 1;
            Phase1Task left = new Phase1Task(from, mid);
            Phase1Task right = new Phase1Task(mid, to);
            left.fork();
            double[][] r = right.compute();
            double[][] l = left.join();
            for (int i = 0; i < denseSize; i++) { l[0][i] += r[0][i]; l[1][i] += r[1][i]; }
            return l;
        }
    }

    /** Phase 1d-a: parallel UPU-List entry collection via ForkJoin. */
    private final class EntryCollectionTask
            extends RecursiveTask<Map<Integer, List<TransactionEntry>>> {
        final int from, to;
        EntryCollectionTask(int from, int to) { this.from = from; this.to = to; }

        @Override
        protected Map<Integer, List<TransactionEntry>> compute() {
            if (to - from <= LEAF_SIZE) return collectEntriesRange(from, to);
            int mid = (from + to) >>> 1;
            EntryCollectionTask left = new EntryCollectionTask(from, mid);
            EntryCollectionTask right = new EntryCollectionTask(mid, to);
            left.fork();
            Map<Integer, List<TransactionEntry>> r = right.compute();
            Map<Integer, List<TransactionEntry>> l = left.join();
            for (Map.Entry<Integer, List<TransactionEntry>> e : r.entrySet())
                l.computeIfAbsent(e.getKey(), x -> new ArrayList<>()).addAll(e.getValue());
            return l;
        }
    }

    /**
     * Phase 3: parallel prefix-based BestFS mining via ForkJoin.
     * Decomposition: single prefix -> direct mine; small range -> per-item tasks;
     * large range -> PTWU-weighted binary split.
     */
    private final class BestFSMiningTask extends RecursiveAction {
        final int rangeStart, rangeEnd;
        BestFSMiningTask(int start, int end) { this.rangeStart = start; this.rangeEnd = end; }

        @Override
        protected void compute() {
            int size = rangeEnd - rangeStart;
            if (size <= 1) {
                if (rangeStart < sortedItems.size()) minePrefix(rangeStart);
                return;
            }
            if (size <= FINE_GRAIN_THRESHOLD) {
                List<BestFSMiningTask> tasks = new ArrayList<>(size);
                for (int i = rangeStart; i < rangeEnd; i++)
                    tasks.add(new BestFSMiningTask(i, i + 1));
                invokeAll(tasks);
            } else {
                int split = findPTWUSplit(rangeStart, rangeEnd);
                invokeAll(new BestFSMiningTask(rangeStart, split),
                          new BestFSMiningTask(split, rangeEnd));
            }
        }

        private void minePrefix(int idx) {
            int item = sortedItems.get(idx);
            UPUList list = singleItemLists.get(item);
            if (list == null || list.entryCount == 0) return;
            if (list.ptwu < collector.getThreshold() - EPSILON) return;
            if (idx + 1 < sortedItems.size()) exploreExtensions(list, idx + 1);
        }
    }

    // =========================================================================
    // Fields - Configuration
    // =========================================================================

    private final String databaseFile;
    private final String profitFile;
    private final String outputFile;
    private final int k;
    private final double minProbability;
    private final boolean parallel;
    private final boolean debug;

    // =========================================================================
    // Fields - Parsed Data
    // =========================================================================

    private List<Transaction> database;
    private Map<Integer, Double> profitTable;
    private int maxItemId;
    private int denseSize;
    private int[] itemIdToDense;
    private int[] denseToItemId;
    private double[] profitCache;

    // =========================================================================
    // Fields - Phase 1 Outputs
    // =========================================================================

    private double[] densePTWU;
    private double[] denseLogComp;
    private Set<Integer> validItems;
    private List<Integer> sortedItems;
    private int[] rankByItemId;
    private Map<Integer, UPUList> singleItemLists;

    // =========================================================================
    // Fields - Mining State & Statistics
    // =========================================================================

    private TopKCollector collector;
    private ForkJoinPool pool;
    private double maxMemory = 0;
    private long totalStartTime, totalEndTime;
    private long phase1Ms, phase2Ms, phase3Ms;

    // =========================================================================
    // Constructor
    // =========================================================================

    public PTK_HUIM_BestFS(String databaseFile, String profitFile, String outputFile,
                         int k, double minProbability, boolean parallel, boolean debug) {
        this.databaseFile = databaseFile;
        this.profitFile = profitFile;
        this.outputFile = outputFile;
        this.k = k;
        this.minProbability = minProbability;
        this.parallel = parallel;
        this.debug = debug;
    }

    // =========================================================================
    // Main Entry Point
    // =========================================================================

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: java PTK_HUIM_BestFS <database> <profits> <k> <minProb> " +
                    "[output] [--no-parallel] [--debug]");
            System.err.println();
            System.err.println("Example:");
            System.err.println("  java PTK_HUIM_BestFS data/chess_db.txt data/chess_profits.txt 100 0.1 --debug");
            System.exit(1);
        }

        String db = args[0], prof = args[1];
        int k = Integer.parseInt(args[2]);
        double minP = Double.parseDouble(args[3]);
        String out = "ptk_huim_bestfs_output.txt";
        boolean par = true, dbg = false;

        for (int i = 4; i < args.length; i++) {
            switch (args[i]) {
                case "--no-parallel": par = false; break;
                case "--debug":      dbg = true;  break;
                default: if (!args[i].startsWith("-")) out = args[i]; break;
            }
        }

        PTK_HUIM_BestFS algo = new PTK_HUIM_BestFS(db, prof, out, k, minP, par, dbg);
        MiningResult result = algo.run();
        algo.writeResults(result);
        algo.printStats(result);
    }

    // =========================================================================
    // Algorithm Execution
    // =========================================================================

    public MiningResult run() throws Exception {
        totalStartTime = System.currentTimeMillis();
        pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());

        try {
            readProfitTable();
            readDatabase();
            checkMemory();
            debugLog("[I/O] Loaded %d transactions, %d items in profit table", database.size(), profitTable.size());

            // Phase 1: Preprocessing
            long p1 = System.currentTimeMillis();
            computePTWU_EP();
            filterAndRankItems();
            buildUPULists();
            phase1Ms = System.currentTimeMillis() - p1;
            checkMemory();
            debugLog("[Phase 1] %d ms | Valid items: %d | UPU-Lists: %d",
                    phase1Ms, validItems.size(), singleItemLists.size());

            // Phase 2: Initialization
            long p2 = System.currentTimeMillis();
            collector = new TopKCollector(k);
            evaluateOneItemsets();
            phase2Ms = System.currentTimeMillis() - p2;
            checkMemory();
            debugLog("[Phase 2] %d ms | Threshold: %.4f", phase2Ms, collector.getThreshold());

            // Phase 3: Mining
            long p3 = System.currentTimeMillis();
            if (parallel) mineParallel(); else mineSequential();
            phase3Ms = System.currentTimeMillis() - p3;
            checkMemory();
            debugLog("[Phase 3] %d ms | Final threshold: %.4f", phase3Ms, collector.getThreshold());

            totalEndTime = System.currentTimeMillis();
            long totalMs = totalEndTime - totalStartTime;
            debugLog("[TOTAL] %d ms", totalMs);

            return new MiningResult(collector.getResults(), totalMs, maxMemory,
                    validItems.size(), singleItemLists.size(), database.size(),
                    phase1Ms, phase2Ms, phase3Ms);
        } finally {
            pool.shutdown();
        }
    }

    // =========================================================================
    // I/O: Read Profit Table
    // =========================================================================

    private void readProfitTable() throws IOException {
        profitTable = new HashMap<>();
        int maxId = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(profitFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.trim().split("[:\\s]+");
                if (p.length >= 2) {
                    try {
                        int id = Integer.parseInt(p[0]);
                        double profit = Double.parseDouble(p[1]);
                        profitTable.put(id, profit);
                        maxId = Math.max(maxId, id);
                    } catch (NumberFormatException e) { /* skip */ }
                }
            }
        }
        this.maxItemId = maxId;

        Set<Integer> allItems = profitTable.keySet();
        denseSize = allItems.size();
        itemIdToDense = new int[maxItemId + 1];
        Arrays.fill(itemIdToDense, -1);
        denseToItemId = new int[denseSize];
        profitCache = new double[denseSize];
        int idx = 0;
        for (int id : allItems) {
            itemIdToDense[id] = idx;
            denseToItemId[idx] = id;
            profitCache[idx] = profitTable.get(id);
            idx++;
        }
    }

    // =========================================================================
    // I/O: Read Database
    // =========================================================================

    private void readDatabase() throws IOException {
        database = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(databaseFile), 32768)) {
            String line; int tid = 1;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                Map<Integer, Integer> qty = new HashMap<>();
                Map<Integer, Double> prob = new HashMap<>();
                for (String token : line.split("\\s+")) {
                    String[] p = token.split(":");
                    if (p.length < 3) continue;
                    try {
                        int id = Integer.parseInt(p[0]);
                        int q = Integer.parseInt(p[1]);
                        double pr = Double.parseDouble(p[2]);
                        if (pr > 0 && pr <= 1.0) { qty.put(id, q); prob.put(id, pr); }
                    } catch (NumberFormatException e) { /* skip */ }
                }
                if (!qty.isEmpty()) database.add(new Transaction(tid++, qty, prob));
            }
        }
    }

    // =========================================================================
    // Phase 1a: Compute PTWU and EP
    // =========================================================================

    private void computePTWU_EP() {
        densePTWU = new double[denseSize];
        denseLogComp = new double[denseSize];
        if (parallel && database.size() > LEAF_SIZE) {
            double[][] result = pool.invoke(new Phase1Task(0, database.size()));
            densePTWU = result[0];
            denseLogComp = result[1];
        } else {
            for (Transaction tx : database)
                processTransactionPhase1(tx, densePTWU, denseLogComp);
        }
    }

    private void processTransactionPhase1(Transaction tx, double[] ptwu, double[] logComp) {
        double ptu = 0.0;
        for (int item : tx.getItems()) {
            if (item < 0 || item > maxItemId) continue;
            int di = itemIdToDense[item];
            if (di < 0) continue;
            if (profitCache[di] > 0) ptu += profitCache[di] * tx.getQuantity(item);
        }
        for (int item : tx.getItems()) {
            if (item < 0 || item > maxItemId) continue;
            int di = itemIdToDense[item];
            if (di < 0) continue;
            ptwu[di] += ptu;
            logComp[di] += logComplement(tx.getProbability(item));
        }
    }

    // =========================================================================
    // Phase 1b + 1c: Filter by EP, Rank by PTWU
    // =========================================================================

    private void filterAndRankItems() {
        validItems = new HashSet<>();
        for (int itemId : profitTable.keySet()) {
            if (itemId < 0 || itemId > maxItemId) continue;
            int di = itemIdToDense[itemId];
            if (di < 0) continue;
            double ep = computeEP(denseLogComp[di]);
            if (ep >= minProbability - EPSILON && densePTWU[di] > 0.0) validItems.add(itemId);
        }

        Integer[] candidates = validItems.toArray(new Integer[0]);
        Arrays.sort(candidates, (a, b) -> {
            int c = Double.compare(densePTWU[itemIdToDense[a]], densePTWU[itemIdToDense[b]]);
            return c != 0 ? c : Integer.compare(a, b);
        });
        sortedItems = Collections.unmodifiableList(Arrays.asList(candidates));

        rankByItemId = new int[maxItemId + 1];
        Arrays.fill(rankByItemId, -1);
        for (int i = 0; i < sortedItems.size(); i++) rankByItemId[sortedItems.get(i)] = i;
    }

    // =========================================================================
    // Phase 1d: Build Single-Item UPU-Lists
    // =========================================================================

    private void buildUPULists() {
        Map<Integer, List<TransactionEntry>> itemEntries = (parallel && database.size() > LEAF_SIZE)
                ? pool.invoke(new EntryCollectionTask(0, database.size()))
                : collectEntriesRange(0, database.size());

        singleItemLists = new HashMap<>();
        for (int item : sortedItems) {
            List<TransactionEntry> entries = itemEntries.get(item);
            if (entries == null || entries.isEmpty()) continue;
            double ptwu = densePTWU[itemIdToDense[item]];
            UPUList list = buildUPUListFromEntries(Collections.singleton(item), entries, ptwu);
            if (list.existentialProbability >= minProbability - EPSILON)
                singleItemLists.put(item, list);
        }
    }

    private Map<Integer, List<TransactionEntry>> collectEntriesRange(int from, int to) {
        Map<Integer, List<TransactionEntry>> result = new HashMap<>();
        for (int i = from; i < to; i++) {
            Transaction tx = database.get(i);
            List<ItemInfo> infos = extractSortedItems(tx);
            if (infos.isEmpty()) continue;
            double[] suffix = computeSuffixSums(infos);
            for (int j = 0; j < infos.size(); j++) {
                ItemInfo info = infos.get(j);
                if (info.logProbability > LOG_ZERO)
                    result.computeIfAbsent(info.itemId, x -> new ArrayList<>())
                            .add(new TransactionEntry(tx.tid, info.utility, suffix[j], info.logProbability));
            }
        }
        return result;
    }

    private List<ItemInfo> extractSortedItems(Transaction tx) {
        List<ItemInfo> items = new ArrayList<>();
        for (int item : tx.getItems()) {
            int rank = (item >= 0 && item < rankByItemId.length) ? rankByItemId[item] : -1;
            if (rank < 0) continue;
            double profit = profitTable.getOrDefault(item, 0.0);
            double prob = tx.getProbability(item);
            if (prob <= 0) continue;
            double logP = Math.max(Math.log(prob), LOG_ZERO);
            items.add(new ItemInfo(item, rank, profit, tx.getQuantity(item), logP));
        }
        if (items.size() > 1) items.sort(null);
        return items;
    }

    private static double[] computeSuffixSums(List<ItemInfo> items) {
        int n = items.size();
        double[] ss = new double[n];
        ss[n - 1] = 0.0;
        for (int i = n - 2; i >= 0; i--) {
            ItemInfo next = items.get(i + 1);
            ss[i] = ss[i + 1] + ((next.profit > 0) ? next.utility : 0.0);
        }
        return ss;
    }

    private static UPUList buildUPUListFromEntries(Set<Integer> itemset,
                                                    List<TransactionEntry> entries, double ptwu) {
        int n = entries.size();
        int[] tids = new int[n]; double[] utils = new double[n];
        double[] rem = new double[n]; double[] probs = new double[n];
        double sumEU = 0, posUB = 0, logComp = 0;

        for (int i = 0; i < n; i++) {
            TransactionEntry e = entries.get(i);
            tids[i] = e.tid; utils[i] = e.utility;
            rem[i] = e.remainingUtility;

            double prob = Math.exp(e.logProbability);
            probs[i] = prob;
            sumEU += e.utility * prob;
            double total = e.utility + e.remainingUtility;
            if (total > 0) posUB += prob * total;

            if (e.logProbability > LOG_ONE_MINUS_EPS) { logComp = LOG_ZERO; }
            else if (logComp >= LOG_ZERO) {
                double l1p = (prob < 0.5) ? Math.log1p(-prob) : Math.log(1.0 - prob);
                logComp += l1p;
                if (logComp < LOG_ZERO) logComp = LOG_ZERO;
            }
        }
        double ep = (logComp <= LOG_ZERO) ? 1.0 : 1.0 - Math.exp(logComp);
        return new UPUList(itemset, tids, utils, rem, probs, n, ptwu, sumEU, ep, posUB);
    }

    // =========================================================================
    // Phase 2: Evaluate 1-Itemsets
    // =========================================================================

    private void evaluateOneItemsets() {
        for (int item : sortedItems) {
            UPUList list = singleItemLists.get(item);
            if (list == null) continue;
            if (list.existentialProbability >= minProbability - EPSILON
                    && list.expectedUtility >= collector.getThreshold() - EPSILON)
                collector.tryCollect(list);
        }
    }

    // =========================================================================
    // Phase 3: Mining
    // =========================================================================

    private void mineParallel() {
        pool.invoke(new BestFSMiningTask(0, sortedItems.size()));
    }

    private void mineSequential() {
        for (int i = 0; i < sortedItems.size(); i++) {
            UPUList prefixList = singleItemLists.get(sortedItems.get(i));
            if (prefixList == null || prefixList.entryCount == 0) continue;
            if (prefixList.ptwu < collector.getThreshold() - EPSILON) continue;
            exploreExtensions(prefixList, i + 1);
        }
    }

    private int findPTWUSplit(int start, int end) {
        double total = 0;
        for (int i = start; i < end; i++) {
            UPUList l = singleItemLists.get(sortedItems.get(i));
            if (l != null) total += l.ptwu;
        }
        if (total == 0) return (start + end) >>> 1;
        double half = total / 2.0, cum = 0;
        int split = start;
        for (int i = start; i < end; i++) {
            UPUList l = singleItemLists.get(sortedItems.get(i));
            if (l != null) cum += l.ptwu;
            if (cum >= half) { split = i + 1; break; }
        }
        return Math.max(start + 1, Math.min(split, end - 1));
    }

    // =========================================================================
    // BestFS Engine: Priority Queue Prefix-Growth with Three-Tier Pruning
    // =========================================================================

    /**
     * Explores all extensions of prefix using Best-First Search (max-PUB priority queue).
     *
     * <p>Instead of recursively descending depth-first, BestFS maintains a max-heap
     * ordered by Positive Upper Bound (PUB) descending. The node with the highest
     * PUB is always expanded next, regardless of depth. This greedy strategy ensures
     * that the most promising patterns are evaluated first, raising the dynamic
     * admission threshold as fast as possible.</p>
     *
     * <p>Stale-node pruning at dequeue: when a node is dequeued, the threshold may
     * have risen since the node was enqueued (due to better patterns found elsewhere).
     * Nodes whose PTWU or PUB now fall below the threshold are discarded immediately,
     * avoiding unnecessary join operations.</p>
     *
     * Three-tier pruning (same correctness as DFS, reordered for deferred EP):
     *   1. PTWU - monotone upper bound (PTWU >= EU for all supersets), O(1)
     *   2. PUB  - tighter upper bound (PUB >= EU, closer to actual EU), O(1)
     *   3. EP   - anti-monotone, deferred: computed ONLY after PTWU+PUB pass, O(count)
     *
     * <p><b>Correctness guarantee:</b> EXACT — all non-pruned nodes are eventually
     * expanded. BestFS and DFS explore the same search space; only traversal order
     * differs. Both produce identical Top-K results.</p>
     *
     * <p><b>Memory:</b> O(frontier size) — may hold many nodes for dense search spaces,
     * compared to O(max depth) for DFS.</p>
     */
    private void exploreExtensions(UPUList prefix, int startIndex) {
        // Max-heap on PUB — most promising patterns expanded first
        PriorityQueue<SearchNode> pq = new PriorityQueue<>(
            Comparator.comparingDouble((SearchNode n) -> n.list.positiveUpperBound).reversed()
        );
        pq.offer(new SearchNode(prefix, startIndex));

        int itemCount = sortedItems.size();

        while (!pq.isEmpty()) {
            SearchNode node = pq.poll();
            UPUList current = node.list;

            // Re-check threshold at dequeue time — prune stale nodes immediately.
            // The threshold may have risen since this node was enqueued because
            // other (higher-PUB) nodes were expanded first and found good patterns.
            double threshold = collector.getThreshold();
            if (current.ptwu < threshold - EPSILON) continue;
            if (current.positiveUpperBound < threshold - EPSILON) continue;

            for (int i = node.startIndex; i < itemCount; i++) {
                int extItem = sortedItems.get(i);
                UPUList extList = singleItemLists.get(extItem);
                if (extList == null) continue;

                // Refresh threshold within the extension loop — enables progressive
                // pruning as the queue yields better patterns
                threshold = collector.getThreshold();

                UPUList joined = joinTwoPointer(current, extList, extItem, threshold);
                if (joined == null || joined.entryCount == 0) continue;

                // Tier 1: PTWU pruning (monotone UB) — O(1), pre-computed in join
                if (joined.ptwu < threshold - EPSILON) continue;
                // Tier 2: PUB pruning (tighter UB) — O(1), pre-computed in join
                if (joined.positiveUpperBound < threshold - EPSILON) continue;

                // Tier 3: EP pruning (anti-monotone) — deferred, O(count)
                // Only computed here because PTWU and PUB already passed.
                double ep = computeEPFromProbs(joined.probabilities, joined.entryCount);
                if (ep < minProbability - EPSILON) continue;
                joined.existentialProbability = ep;

                // Admission: only collect patterns with EU >= threshold.
                // Threshold starts at 0 — effectively EU >= 0 until heap fills.
                if (joined.expectedUtility >= threshold - EPSILON) {
                    collector.tryCollect(joined);
                    threshold = collector.getThreshold();
                }

                // Enqueue for future expansion — PQ ordering ensures highest-PUB
                // nodes are expanded first, maximizing threshold growth
                if (i + 1 < itemCount) {
                    pq.offer(new SearchNode(joined, i + 1));
                }
            }
        }
    }

    // =========================================================================
    // Two-Pointer UPU-List Join with Inline EU/PUB Aggregation (EP Deferred)
    // =========================================================================

    /**
     * Joins prefix UPU-List with single-item extension UPU-List.
     * O(|L1| + |L2|) two-pointer merge computing EU and PUB inline.
     *
     * <p><b>Optimization:</b> Probabilities are multiplied directly (prob1 × prob2)
     * instead of computing exp(logP1 + logP2), eliminating all Math.exp() calls.
     * EP computation is deferred — not computed here. The returned UPUList has
     * EP = -1.0 (sentinel). Caller must invoke {@link #computeEPFromProbs}
     * after PTWU/PUB pruning passes.</p>
     */
    private static UPUList joinTwoPointer(UPUList list1, UPUList list2,
                                           int extensionItem, double threshold) {
        double joinedPTWU = Math.min(list1.ptwu, list2.ptwu);
        if (joinedPTWU < threshold - EPSILON) return null;

        int maxCount = Math.min(list1.entryCount, list2.entryCount);
        int[] tids = new int[maxCount]; double[] utils = new double[maxCount];
        double[] rems = new double[maxCount]; double[] probs = new double[maxCount];
        double sumEU = 0, posUB = 0;
        int count = 0, i = 0, j = 0;

        int[] t1 = list1.transactionIds, t2 = list2.transactionIds;
        double[] u1 = list1.utilities, u2 = list2.utilities;
        double[] r1 = list1.remainingUtilities, r2 = list2.remainingUtilities;
        double[] p1 = list1.probabilities, p2 = list2.probabilities;

        while (i < list1.entryCount && j < list2.entryCount) {
            if (t1[i] == t2[j]) {
                double u = u1[i] + u2[j];
                double r = Math.min(r1[i], r2[j]);
                double prob = p1[i] * p2[j];
                tids[count] = t1[i]; utils[count] = u;
                rems[count] = r; probs[count] = prob; count++;

                sumEU += u * prob;
                double total = u + r;
                if (total > 0) posUB += prob * total;

                i++; j++;
            } else if (t1[i] < t2[j]) { i++; } else { j++; }
        }

        if (count == 0) return null;
        int sz = list1.itemset.size() + 1;
        Set<Integer> itemset = new HashSet<>((sz * 4) / 3 + 1);
        itemset.addAll(list1.itemset);
        itemset.add(extensionItem);
        // EP = -1.0 sentinel: deferred until after PTWU/PUB pruning
        return new UPUList(itemset, tids, utils, rems, probs, count,
                joinedPTWU, sumEU, -1.0, posUB);
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /**
     * Computes Existential Probability from pre-stored probability values.
     * Used for deferred EP computation after PTWU/PUB pruning passes.
     *
     * <p>EP = 1 - Π(1 - prob_i), computed in log-space for numerical stability:
     * logComp = Σ log(1 - prob_i), then EP = 1 - exp(logComp).</p>
     */
    private static double computeEPFromProbs(double[] probs, int count) {
        double logComp = 0;
        for (int i = 0; i < count; i++) {
            double prob = probs[i];
            if (prob >= 1.0 - EPSILON) return 1.0;
            if (logComp <= LOG_ZERO) break;
            double l1p = (prob < 0.5) ? Math.log1p(-prob) : Math.log(1.0 - prob);
            logComp += l1p;
            if (logComp < LOG_ZERO) logComp = LOG_ZERO;
        }
        return (logComp <= LOG_ZERO) ? 1.0 : 1.0 - Math.exp(logComp);
    }

    private static double logComplement(double probability) {
        if (probability <= 0) return 0.0;
        if (probability >= 1.0) return LOG_ZERO;
        return (probability < 0.5) ? Math.log1p(-probability) : Math.log(1.0 - probability);
    }

    private static double computeEP(double logComp) {
        return (logComp <= LOG_ZERO) ? 1.0 : 1.0 - Math.exp(logComp);
    }

    private void debugLog(String format, Object... args) {
        if (debug) System.err.printf(format + "%n", args);
    }

    private void checkMemory() {
        Runtime rt = Runtime.getRuntime();
        maxMemory = Math.max(maxMemory, (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0));
    }

    // =========================================================================
    // Output
    // =========================================================================

    private void writeResults(MiningResult result) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("=================================================\n");
        sb.append(String.format("PTK-HUIM-BestFS Results: Top-%d Patterns%n", result.patterns.size()));
        sb.append(String.format("Parameters: k=%d, minProb=%.4f, parallel=%s%n", k, minProbability, parallel));
        sb.append("=================================================\n");
        sb.append(String.format(Locale.US, "%-6s %-40s %-15s %-15s%n", "Rank", "Pattern", "Expected Util", "Exist Prob"));
        sb.append("-------------------------------------------------\n");

        int rank = 1;
        for (Pattern p : result.patterns) {
            sb.append(String.format(Locale.US, "%-6d %-40s %-15.4f %-15.6f%n",
                    rank++, p.sortedItems.toString(), p.expectedUtility, p.existentialProbability));
        }

        sb.append("=================================================\n");
        sb.append(String.format(Locale.US, "Execution time: %.3f seconds%n", result.executionTimeMs / 1000.0));
        sb.append(String.format("Patterns found: %d%n", result.patterns.size()));
        sb.append(String.format(Locale.US, "Memory used: %.2f MB%n", result.memoryUsedMB));
        sb.append("=================================================\n");

        try (BufferedWriter w = new BufferedWriter(new FileWriter(outputFile))) { w.write(sb.toString()); }
        System.out.print(sb);
    }

    private void printStats(MiningResult result) {
        System.err.println("============ PTK-HUIM-BestFS STATS =============");
        System.err.printf(" Total time:     %d ms%n", result.executionTimeMs);
        System.err.printf("   Phase 1:      %d ms (preprocessing)%n", result.phase1Ms);
        System.err.printf("   Phase 2:      %d ms (initialization)%n", result.phase2Ms);
        System.err.printf("   Phase 3:      %d ms (mining)%n", result.phase3Ms);
        System.err.printf(Locale.US, " Memory:         %.2f MB%n", result.memoryUsedMB);
        System.err.printf(" Patterns found: %d%n", result.patterns.size());
        System.err.printf(Locale.US, " Final threshold:%.4f%n", collector.getThreshold());
        System.err.printf(" Database:       %d transactions%n", result.databaseSize);
        System.err.printf(" Valid items:    %d%n", result.validItemCount);
        System.err.printf(" UPU-Lists:      %d%n", result.upuListCount);
        System.err.printf(" Parallel:       %s%n", parallel);
        System.err.println("================================================");
    }
}