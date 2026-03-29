package algorithms;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * PTK-HUIM (DFS): Top-K High-Utility Itemset Mining on Uncertain Databases with both positive and negative utilities
 *
 *
 * <h3>Algorithm Configuration</h3>
 * <ul>
 *   <li><b>Search strategy:</b>  Depth-First Search (DFS) prefix-growth</li>
 * </ul>
 *
 * <h3>Three-Phase Pipeline</h3>
 * <ol>
 *   <li><b>Phase 1 - Preprocessing:</b> Compute PTWU, filter items, rank by PTWU,
 *       build single-item UPU-Lists</li>
 *   <li><b>Phase 2 - Initialization:</b> Evaluate 1-itemsets, seed the Top-K collector</li>
 *   <li><b>Phase 3 - Mining:</b> Recursive DFS prefix-growth with two-tier pruning:
 *       PTWU, PUB</li>
 * </ol>
 *
 * <h3>Input Format</h3>
 * <pre>
 *   Database: "itemId:quantity:probability itemId:quantity:probability ..." per line
 *   Profits:  "itemId profit" per line (negative profits allowed)
 * </pre>
 *
 * @author Dang Nguyen Le
 */
public class PTK_HUIM_DFS {

    // =========================================================================
    // Constants
    // =========================================================================

    /** Floating-point comparison tolerance for threshold checks. */
    private static final double EPSILON = 1e-10;

    /** Log-space floor to prevent denormalized underflow. */
    private static final double LOG_ZERO = -700.0;

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
     * Immutable after construction - safe for concurrent reads in Phase 3.
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
        final double positiveUpperBound;

        UPUList(Set<Integer> itemset, int[] tids, double[] utils, double[] remaining,
                double[] probs, int count, double ptwu, double eu, double pub) {
            this.itemset = itemset;
            this.transactionIds = tids;
            this.utilities = utils;
            this.remainingUtilities = remaining;
            this.probabilities = probs;
            this.entryCount = count;
            this.ptwu = ptwu;
            this.expectedUtility = eu;
            this.positiveUpperBound = pub;
        }

        @Override
        public String toString() {
            return String.format("UPUList{items=%s, entries=%d, EU=%.4f}",
                    itemset, entryCount, expectedUtility);
        }
    }

    // =========================================================================
    // Inner Classes - Result Models
    // =========================================================================

    /**
     * Discovered high-utility pattern.
     * Ordered by EU ascending (min-heap). Equality by itemset identity only.
     */
    public static final class Pattern implements Comparable<Pattern> {
        public final Set<Integer> items;
        final double expectedUtility;
        private final List<Integer> sortedItems;

        Pattern(Set<Integer> items, double eu) {
            this.items = items;
            this.expectedUtility = eu;
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
            return String.format("Pattern{items=%s, EU=%.4f}", sortedItems, expectedUtility);
        }
    }

    /**
     * Encapsulates the complete mining result for clean return from run().
     */
    public static final class MiningResult {
        public final List<Pattern> patterns;
        final long executionTimeMs;
        public final double memoryUsedMB;
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
                        Pattern updated = new Pattern(itemset, eu);
                        heap.add(updated);
                        index.put(itemset, updated);
                        updateThreshold();
                        return true;
                    }
                    return false;
                }

                Pattern newPattern = new Pattern(itemset, eu);
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

    /**
     * Phase 3: parallel prefix-based DFS mining via ForkJoin.
     * Decomposition: single prefix -> direct mine; small range -> per-item tasks;
     * large range -> PTWU-weighted binary split.
     */
    private final class DFSMiningTask extends RecursiveAction {
        final int rangeStart, rangeEnd;
        DFSMiningTask(int start, int end) { this.rangeStart = start; this.rangeEnd = end; }

        @Override
        protected void compute() {
            int size = rangeEnd - rangeStart;
            if (size <= 1) {
                if (rangeStart < sortedItems.size()) minePrefix(rangeStart);
                return;
            }
            if (size <= FINE_GRAIN_THRESHOLD) {
                List<DFSMiningTask> tasks = new ArrayList<>(size);
                for (int i = rangeStart; i < rangeEnd; i++)
                    tasks.add(new DFSMiningTask(i, i + 1));
                invokeAll(tasks);
            } else {
                int split = findPTWUSplit(rangeStart, rangeEnd);
                invokeAll(new DFSMiningTask(rangeStart, split),
                          new DFSMiningTask(split, rangeEnd));
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

    public PTK_HUIM_DFS(String databaseFile, String profitFile, String outputFile,
                         int k, boolean parallel, boolean debug) {
        this.databaseFile = databaseFile;
        this.profitFile = profitFile;
        this.outputFile = outputFile;
        this.k = k;
        this.parallel = parallel;
        this.debug = debug;
    }

    // =========================================================================
    // Main Entry Point
    // =========================================================================

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java PTK_HUIM_DFS <database> <profits> <k> " +
                    "[output] [--no-parallel] [--debug]");
            System.err.println();
            System.err.println("Example:");
            System.err.println("  java PTK_HUIM_DFS data/chess_db.txt data/chess_profits.txt 100 --debug");
            System.exit(1);
        }

        String db = args[0], prof = args[1];
        int k = Integer.parseInt(args[2]);
        String out = "ptk_huim_dfs_output.txt";
        boolean par = true, dbg = false;

        for (int i = 3; i < args.length; i++) {
            switch (args[i]) {
                case "--no-parallel": par = false; break;
                case "--debug":      dbg = true;  break;
                default: if (!args[i].startsWith("-")) out = args[i]; break;
            }
        }

        PTK_HUIM_DFS algo = new PTK_HUIM_DFS(db, prof, out, k, par, dbg);
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
            computePTWU();
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
    // Phase 1a: Compute PTWU
    // =========================================================================

    private void computePTWU() {
        densePTWU = new double[denseSize];
        for (Transaction tx : database)
            processTransactionPhase1(tx, densePTWU);
    }

    private void processTransactionPhase1(Transaction tx, double[] ptwu) {
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
        }
    }

    // =========================================================================
    // Phase 1b + 1c: Filter Items, Rank by PTWU
    // =========================================================================

    private void filterAndRankItems() {
        validItems = new HashSet<>();
        for (int itemId : profitTable.keySet()) {
            if (itemId < 0 || itemId > maxItemId) continue;
            int di = itemIdToDense[itemId];
            if (di < 0) continue;
            if (densePTWU[di] > 0.0) validItems.add(itemId);
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
        Map<Integer, List<TransactionEntry>> itemEntries = collectEntriesRange(0, database.size());

        singleItemLists = new HashMap<>();
        for (int item : sortedItems) {
            List<TransactionEntry> entries = itemEntries.get(item);
            if (entries == null || entries.isEmpty()) continue;
            double ptwu = densePTWU[itemIdToDense[item]];
            UPUList list = buildUPUListFromEntries(Collections.singleton(item), entries, ptwu);
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
        double sumEU = 0, posUB = 0;

        for (int i = 0; i < n; i++) {
            TransactionEntry e = entries.get(i);
            tids[i] = e.tid; utils[i] = e.utility;
            rem[i] = e.remainingUtility;

            double prob = Math.exp(e.logProbability);
            probs[i] = prob;
            sumEU += e.utility * prob;
            double total = e.utility + e.remainingUtility;
            if (total > 0) posUB += prob * total;
        }
        return new UPUList(itemset, tids, utils, rem, probs, n, ptwu, sumEU, posUB);
    }

    // =========================================================================
    // Phase 2: Evaluate 1-Itemsets
    // =========================================================================

    private void evaluateOneItemsets() {
        for (int item : sortedItems) {
            UPUList list = singleItemLists.get(item);
            if (list == null) continue;
            if (list.expectedUtility >= collector.getThreshold() - EPSILON)
                collector.tryCollect(list);
        }
    }

    // =========================================================================
    // Phase 3: Mining
    // =========================================================================

    private void mineParallel() {
        pool.invoke(new DFSMiningTask(0, sortedItems.size()));
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
    // DFS Engine: Recursive Prefix-Growth with Two-Tier Pruning
    // =========================================================================

    /**
     * Recursively explores all extensions of prefix in PTWU-ascending order.
     *
     * Two-tier pruning:
     *   1. PTWU - monotone upper bound (PTWU >= EU for all supersets), O(1)
     *   2. PUB  - tighter upper bound (PUB >= EU, closer to actual EU), O(1)
     */
    private void exploreExtensions(UPUList prefix, int startIndex) {
        double threshold = collector.getThreshold();
        if (prefix.ptwu < threshold - EPSILON) return;
        if (prefix.positiveUpperBound < threshold - EPSILON) return;

        int itemCount = sortedItems.size();
        for (int i = startIndex; i < itemCount; i++) {
            int extItem = sortedItems.get(i);
            UPUList extList = singleItemLists.get(extItem);
            if (extList == null) continue;
            threshold = collector.getThreshold();

            if (extList.ptwu < threshold - EPSILON) continue;
            UPUList joined = joinTwoPointer(prefix, extList, extItem, threshold);
            if (joined == null || joined.entryCount == 0) continue;

            // Tier 1: PTWU pruning (monotone UB) — O(1), pre-computed in join
            if (joined.ptwu < threshold - EPSILON) continue;
            // Tier 2: PUB pruning (tighter UB) — O(1), pre-computed in join
            if (joined.positiveUpperBound < threshold - EPSILON) continue;

            // Admission: only collect patterns with EU >= threshold.
            // Threshold starts at 0 — effectively EU >= 0 until heap fills.
            if (joined.expectedUtility >= threshold - EPSILON) {
                collector.tryCollect(joined);
                threshold = collector.getThreshold();
            }

            exploreExtensions(joined, i + 1);
        }
    }

    // =========================================================================
    // Two-Pointer UPU-List Join with Inline EU/PUB Aggregation
    // =========================================================================

    /**
     * Joins prefix UPU-List with single-item extension UPU-List.
     * O(|L1| + |L2|) two-pointer merge computing EU and PUB inline.
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
        return new UPUList(itemset, tids, utils, rems, probs, count,
                joinedPTWU, sumEU, posUB);
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

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
        sb.append(String.format("PTK-HUIM-DFS Results: Top-%d Patterns%n", result.patterns.size()));
        sb.append(String.format("Parameters: k=%d, parallel=%s%n", k, parallel));
        sb.append("=================================================\n");
        sb.append(String.format(Locale.US, "%-6s %-40s %-15s%n", "Rank", "Pattern", "Expected Util"));
        sb.append("-------------------------------------------------\n");

        int rank = 1;
        for (Pattern p : result.patterns) {
            sb.append(String.format(Locale.US, "%-6d %-40s %-15.4f%n",
                    rank++, p.sortedItems.toString(), p.expectedUtility));
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
        System.err.println("============= PTK-HUIM-DFS STATS ==============");
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