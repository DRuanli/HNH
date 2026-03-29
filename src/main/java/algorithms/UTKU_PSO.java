package algorithms;

import java.io.*;
import java.util.*;

/**
 * UTKU-PSO: Uncertain Top-K Utility Particle Swarm Optimization
 *
 * Adapted from TKU-PSO (Carstensen & Lin, 2024) for:
 *   - Uncertain transaction databases (items have occurrence probabilities)
 *   - Positive AND negative item profits
 *
 * Key changes from TKU-PSO:
 *   1. Fitness = Expected Utility: EU(X) = Σ_T P(X,T) × u(X,T)
 *   2. Pruning uses PTWU (only positive-profit items in transaction weight)
 *   3. MSF initialized to 0 (only EU >= 0 patterns accepted initially)
 *   4. Fitness estimation adapted for probability-weighted contributions
 *   5. Bit-clear uses PTWU < MSF
 *
 * Input format:
 *   Database file: one transaction per line
 *     itemId:quantity:probability itemId:quantity:probability ...
 *   Profit file: one item per line
 *     itemId profit
 *
 * @author Dang Nguyen Le
 */
public class UTKU_PSO {

    // =========================================================================
    // Internal Data Structures
    // =========================================================================

    /**
     * Represents one item occurrence in a transaction.
     * Stores item ID, utility (profit × quantity), and probability.
     */
    private static class TransEntry implements Comparable<TransEntry> {
        final int item;          // renamed item ID (1-based)
        final double utility;    // profit(item) × quantity — can be negative
        final double probability;// P(item, T) ∈ (0, 1]

        TransEntry(int item, double utility, double probability) {
            this.item = item;
            this.utility = utility;
            this.probability = probability;
        }

        @Override
        public int compareTo(TransEntry o) {
            return Integer.compare(this.item, o.item);
        }
    }

    /**
     * Stores per-item precomputed statistics.
     */
    private static class Item implements Comparable<Item> {
        final int item;           // renamed item ID (1-based)
        BitSet TIDS;              // TID-set: transactions containing this item
        double ptwu;              // PTWU (Positive Transaction-Weighted Utility)
        double totalEU = 0.0;     // EU of 1-itemset: Σ_T P(i,T) × u(i,T)
        double avgEU;             // average expected contribution per transaction
        double maxEU = 0.0;       // maximum |P(i,T) × u(i,T)| across transactions

        Item(int item) {
            this.item = item;
            this.TIDS = new BitSet();
        }

        @Override
        public int compareTo(Item o) {
            // Sort by totalEU ascending (for TreeSet ordering)
            if (this.totalEU < o.totalEU) return -1;
            if (this.totalEU > o.totalEU) return 1;
            return Integer.compare(this.item, o.item);
        }

        @Override
        public String toString() {
            return String.valueOf(item);
        }
    }

    /**
     * Represents a particle (candidate solution) in the PSO population.
     */
    private static class Particle implements Comparable<Particle> {
        BitSet X;                 // encoding vector (itemset)
        double fitness;           // EU of the itemset
        double estFitness;        // estimated fitness (per-item sum, before × support)

        Particle(int size) {
            this.X = new BitSet(size);
            this.fitness = Double.NEGATIVE_INFINITY;
        }

        Particle(BitSet bitset, double fitness) {
            this.X = (BitSet) bitset.clone();
            this.fitness = fitness;
        }

        @Override
        public int compareTo(Particle o) {
            // For TreeSet ordering — must never return 0 for different particles
            if (this.fitness < o.fitness) return -1;
            if (this.fitness > o.fitness) return 1;
            // Tiebreak by itemset hash to avoid TreeSet deduplication
            int h1 = this.X.hashCode();
            int h2 = o.X.hashCode();
            if (h1 != h2) return Integer.compare(h1, h2);
            // Last resort: compare bitsets lexicographically
            for (int i = Math.max(this.X.length(), o.X.length()); i >= 0; i--) {
                boolean b1 = this.X.get(i);
                boolean b2 = o.X.get(i);
                if (b1 != b2) return b1 ? 1 : -1;
            }
            return 0; // truly identical
        }
    }

    /**
     * Maintains the top-k solution set.
     * Sorted in descending EU order for efficient RWS.
     */
    private class Solutions {
        final int capacity;
        // Reversed comparator: highest EU first (for RWS iteration)
        TreeSet<Particle> sol = new TreeSet<>(Comparator.reverseOrder());

        Solutions(int k) {
            this.capacity = k;
        }

        /**
         * Attempts to add a pattern to the top-k set.
         * If full, evicts the weakest (lowest EU) pattern.
         */
        void add(Particle p) {
            if (sol.size() == capacity) {
                Particle weakest = sol.last(); // lowest EU
                euSum -= weakest.fitness;
                sol.remove(weakest);
            }
            // Check if this particle is the new fittest
            if (!sol.isEmpty()) {
                runRWS = (p.fitness > sol.first().fitness) ? false : runRWS;
            }
            sol.add(p);
            euSum += p.fitness;
            newSolution = true;
            if (sol.size() == capacity) {
                minSolutionFitness = sol.last().fitness; // MSF = EU of k-th best
            }
        }

        TreeSet<Particle> getSol() { return sol; }
        int getSize() { return sol.size(); }
    }

    // =========================================================================
    // Inner Classes - Result Model
    // =========================================================================

    /**
     * Encapsulates the complete mining result for clean return from run().
     * Mirrors PTK_HUIM's MiningResult for consistent experiment API.
     */
    public static final class MiningResult {
        public final int patternCount;
        public final double memoryUsedMB;
        public final long executionTimeMs;
        public final List<PatternEntry> patterns;

        MiningResult(int patternCount, double memoryMB, long execMs,
                     List<PatternEntry> patterns) {
            this.patternCount = patternCount;
            this.memoryUsedMB = memoryMB;
            this.executionTimeMs = execMs;
            this.patterns = patterns;
        }
    }

    /**
     * One discovered pattern — itemset + EU, for experiment comparison.
     */
    public static final class PatternEntry {
        public final Set<Integer> items;
        public final double expectedUtility;

        PatternEntry(Set<Integer> items, double eu) {
            this.items = items;
            this.expectedUtility = eu;
        }
    }

    // =========================================================================
    // Fields
    // =========================================================================

    // Database
    private List<TransEntry[]> database = new ArrayList<>(); // pruned database
    private Map<Integer, Double> profitTable = new HashMap<>(); // itemId → profit

    // PSO state
    private Particle gBest;
    private Particle[] pBest;
    private Particle[] population;
    private ArrayList<Item> validItems = new ArrayList<>(); // valid items (PTWU ≥ CEU)
    private HashSet<BitSet> explored;
    private HashMap<Integer, Integer> itemNamesRev = new HashMap<>(); // renamed → original

    // Estimation
    private double std;           // deviation for fitness estimation
    private int lowEst = 0;       // underestimates count
    private int highEst = 0;      // overestimates count

    // Solution tracking
    private double minSolutionFitness = 0; // MSF starts at 0 (only EU >= 0 patterns accepted)
    private Solutions solutions;
    private boolean newSolution = false;
    private TreeSet<Item> sizeOneItemsets;
    private boolean runRWS = true;
    private double euSum = 0.0;   // sum of EU in current top-k (for RWS)
    private double ptwuSum = 0.0; // sum of PTWU of all valid items (for RWS)
    private int maxTransactionLength = 0;

    // Algorithm parameters
    private String databaseFile;
    private String profitFile;
    private String outputFile;
    private int pop_size = 20;
    private int iterations = 10000;
    private int k = 100;

    // Stats
    private double maxMemory;
    private long startTimestamp;
    private long endTimestamp;

    // =========================================================================
    // Constructor
    // =========================================================================

    public UTKU_PSO(String databaseFile, String profitFile, String outputFile,
                    int k, int pop_size, int iterations) {
        this.databaseFile = databaseFile;
        this.profitFile = profitFile;
        this.outputFile = outputFile;
        this.k = k;
        this.pop_size = pop_size;
        this.iterations = iterations;
    }

    // =========================================================================
    // Main Entry Point
    // =========================================================================

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.err.println("Usage: java UTKU_PSO <database> <profits> <k> " +
                    "[pop_size] [iterations] [output]");
            System.err.println("Example: java UTKU_PSO data/chess_database.txt data/chess_profits.txt 100");
            System.exit(1);
        }

        String dbFile = args[0];
        String profitFile = args[1];
        int k = Integer.parseInt(args[2]);
        int popSize = (args.length > 3) ? Integer.parseInt(args[3]) : 20;
        int iters = (args.length > 4) ? Integer.parseInt(args[4]) : 10000;
        String outFile = (args.length > 5) ? args[5] : "utku_pso_output.txt";

        UTKU_PSO algorithm = new UTKU_PSO(dbFile, profitFile, outFile, k, popSize, iters);
        MiningResult result = algorithm.run();
        algorithm.printStats();
        System.out.printf("Returned: %d patterns, %.2f MB%n",
                result.patternCount, result.memoryUsedMB);
    }

    // =========================================================================
    // Algorithm Execution
    // =========================================================================

    /**
     * Runs the complete UTKU-PSO algorithm.
     *
     * @return MiningResult with pattern count, memory, time, and discovered patterns
     */
    public MiningResult run() throws IOException {
        maxMemory = 0;
        startTimestamp = System.currentTimeMillis();

        // Phase 1: Read profit table and initialize database
        readProfitTable();
        init();
        solutions = new Solutions(k);
        checkMemory();

        System.out.println("Valid items (PTWU filtered): " + validItems.size());

        // Precompute item statistics for estimation
        sizeOneItemsets = new TreeSet<>();
        std = 0.0;
        for (Item item : validItems) {
            int support = item.TIDS.cardinality();
            if (support > 0) {
                item.avgEU = item.totalEU / support;
            }
            std += item.maxEU - Math.abs(item.avgEU);
            sizeOneItemsets.add(item);
            ptwuSum += item.ptwu;
        }

        explored = new HashSet<>();
        explored.add(new BitSet(validItems.size()));

        if (validItems.size() != 0) {
            std = std / validItems.size();
            if (std < 0) std = 0; // safety: deviation cannot be negative

            generatePop();       // Phase 2: Initialize population
            fillSolutions();     // Fill solution set with remaining 1-itemsets
            List<Double> probRange = rouletteTopK();

            // Phase 3: Main PSO loop
            for (int i = 0; i < iterations; i++) {
                runRWS = true;
                update();

                if (i > 1 && runRWS) {
                    if (newSolution) {
                        probRange = rouletteTopK();
                        newSolution = false;
                    }
                    if (!probRange.isEmpty()) {
                        int pos = rouletteSelect(probRange);
                        selectGBest(pos);
                    }
                }

                // Tighten deviation every 25 iterations
                if (i % 25 == 0 && highEst > 0 && i > 0 && std > 1e-6) {
                    std = ((double) lowEst / highEst < 0.01) ? std / 2.0 : std;
                }
            }
        }

        endTimestamp = System.currentTimeMillis();
        checkMemory();
        writeOutput();

        return buildResult();
    }

    /**
     * Builds the MiningResult from current solution state.
     * Called after run() completes — always safe, even if writeOutput() fails.
     */
    private MiningResult buildResult() {
        List<PatternEntry> patterns = new ArrayList<>();
        for (Particle p : solutions.getSol()) {
            Set<Integer> items = new LinkedHashSet<>();
            for (int i = p.X.nextSetBit(0); i != -1; i = p.X.nextSetBit(i + 1)) {
                items.add(itemNamesRev.get(i));
            }
            patterns.add(new PatternEntry(items, p.fitness));
        }
        return new MiningResult(
                solutions.getSize(),
                maxMemory,
                endTimestamp - startTimestamp,
                patterns
        );
    }

    // =========================================================================
    // Phase 1: Initialization (Database Reading + Pruning)
    // =========================================================================

    /**
     * Reads the profit table from file.
     * Format: "itemId profit" per line (space or colon separated).
     */
    private void readProfitTable() throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(profitFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("[:\\s]+");
                if (parts.length >= 2) {
                    try {
                        int itemId = Integer.parseInt(parts[0]);
                        double profit = Double.parseDouble(parts[1]);
                        profitTable.put(itemId, profit);
                    } catch (NumberFormatException e) {
                        // skip malformed line
                    }
                }
            }
        }
    }

    /**
     * Reads the uncertain database, computes PTWU/EU for each item,
     * prunes items, and builds the internal database representation.
     *
     * Two database scans:
     *   Scan 1: Compute PTWU, EU for each 1-itemset
     *   Scan 2: Prune items and build database matrix
     */
    private void init() throws IOException {
        // =====================================================================
        // Scan 1: Compute PTWU, EU of each item
        // =====================================================================
        Map<Integer, Double> itemPTWU = new HashMap<>();
        Map<Integer, Double> itemEU = new HashMap<>();
        Map<Integer, Double> itemMaxEU = new HashMap<>();   // max |P×u| per transaction

        try (BufferedReader reader = new BufferedReader(new FileReader(databaseFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] tokens = line.split("\\s+");

                // First pass over transaction: compute PTU(T)
                double ptu = 0.0;
                List<int[]> parsedItems = new ArrayList<>();      // [itemId, quantity]
                List<Double> parsedProbs = new ArrayList<>();

                for (String token : tokens) {
                    String[] parts = token.split(":");
                    if (parts.length < 3) continue;
                    try {
                        int itemId = Integer.parseInt(parts[0]);
                        int qty = Integer.parseInt(parts[1]);
                        double prob = Double.parseDouble(parts[2]);
                        if (prob <= 0.0 || prob > 1.0) continue;

                        double profit = profitTable.getOrDefault(itemId, 0.0);
                        double utility = profit * qty;

                        // PTU only counts positive-profit items
                        if (profit > 0) {
                            ptu += utility;
                        }

                        parsedItems.add(new int[]{itemId, qty});
                        parsedProbs.add(prob);
                    } catch (NumberFormatException e) {
                        // skip malformed token
                    }
                }

                // Second pass: distribute PTU to PTWU, accumulate EU
                for (int i = 0; i < parsedItems.size(); i++) {
                    int itemId = parsedItems.get(i)[0];
                    int qty = parsedItems.get(i)[1];
                    double prob = parsedProbs.get(i);
                    double profit = profitTable.getOrDefault(itemId, 0.0);
                    double utility = profit * qty;

                    // PTWU accumulation
                    itemPTWU.merge(itemId, ptu, Double::sum);

                    // EU accumulation: P(i,T) × u(i,T)
                    double contribution = prob * utility;
                    itemEU.merge(itemId, contribution, Double::sum);

                    // Max |expected contribution| per transaction
                    double absContrib = Math.abs(contribution);
                    itemMaxEU.merge(itemId, absContrib, Math::max);
                }
            }
        }

        // =====================================================================
        // Compute CEU (Critical Expected Utility) = k-th largest EU among items
        // =====================================================================
        List<Map.Entry<Integer, Double>> euList = new ArrayList<>(itemEU.entrySet());
        euList.sort((a, b) -> Double.compare(b.getValue(), a.getValue())); // descending EU
        double ceu = (k <= euList.size()) ? euList.get(k - 1).getValue() : 0.0;

        // Use CEU as minimum utility threshold for PTWU pruning
        // Items with PTWU < CEU cannot be in any top-k pattern
        double minUtilThreshold = Math.max(0.0, ceu); // don't use negative threshold for PTWU pruning
        System.out.println("CEU (k-th largest 1-item EU): " + ceu);
        System.out.println("PTWU pruning threshold: " + minUtilThreshold);

        // =====================================================================
        // Build valid item set: PTWU ≥ threshold
        // =====================================================================
        // Rename items from 1 to #validItems (sorted by EU descending for better init)
        HashMap<Integer, Integer> itemNames = new HashMap<>(); // original → renamed
        int name = 1;

        for (Map.Entry<Integer, Double> e : euList) {
            int origItem = e.getKey();
            double ptwu = itemPTWU.getOrDefault(origItem, 0.0);

            if (ptwu >= minUtilThreshold) {
                itemNames.put(origItem, name);
                itemNamesRev.put(name, origItem);

                Item item = new Item(name);
                item.ptwu = ptwu;
                item.totalEU = itemEU.getOrDefault(origItem, 0.0);
                item.maxEU = itemMaxEU.getOrDefault(origItem, 0.0);
                validItems.add(item);
                name++;
            }
        }

        // =====================================================================
        // Scan 2: Prune and build database matrix
        // =====================================================================
        try (BufferedReader reader = new BufferedReader(new FileReader(databaseFile))) {
            String line;
            int tid = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] tokens = line.split("\\s+");
                List<TransEntry> transaction = new ArrayList<>();

                for (String token : tokens) {
                    String[] parts = token.split(":");
                    if (parts.length < 3) continue;
                    try {
                        int origItem = Integer.parseInt(parts[0]);
                        int qty = Integer.parseInt(parts[1]);
                        double prob = Double.parseDouble(parts[2]);

                        if (itemNames.containsKey(origItem)) {
                            int renamedItem = itemNames.get(origItem);
                            double profit = profitTable.getOrDefault(origItem, 0.0);
                            double utility = profit * qty;

                            transaction.add(new TransEntry(renamedItem, utility, prob));

                            // Update item TID-set and max EU
                            Item itemObj = validItems.get(renamedItem - 1);
                            itemObj.TIDS.set(tid);
                            double absContrib = Math.abs(prob * utility);
                            itemObj.maxEU = Math.max(itemObj.maxEU, absContrib);
                        }
                    } catch (NumberFormatException e) {
                        // skip
                    }
                }

                if (!transaction.isEmpty()) {
                    Collections.sort(transaction); // sort by renamed item ID
                    maxTransactionLength = Math.max(maxTransactionLength, transaction.size());
                    TransEntry[] trans = transaction.toArray(new TransEntry[0]);
                    database.add(trans);
                    tid++;
                }
            }
        }
    }

    // =========================================================================
    // Phase 2: Population Initialization
    // =========================================================================

    /**
     * Initializes the population.
     * First particles get best 1-itemsets; remainder use RWS.
     */
    private void generatePop() {
        List<Double> rouletteProbabilities = (validItems.size() < pop_size) ? roulettePTWU() : null;
        population = new Particle[pop_size];
        pBest = new Particle[pop_size];

        for (int i = 0; i < pop_size; i++) {
            Particle p = new Particle(validItems.size());

            if (!sizeOneItemsets.isEmpty()) {
                // Initialize to next best 1-itemset (highest EU)
                p.X.set(sizeOneItemsets.pollLast().item);
            } else {
                // RWS initialization (random itemset size, PTWU-weighted selection)
                int numItems = (int) (Math.random() * maxTransactionLength) + 1;
                int added = 0;
                while (added < numItems) {
                    int pos = rouletteSelect(rouletteProbabilities);
                    if (!p.X.get(validItems.get(pos).item)) {
                        p.X.set(validItems.get(pos).item);
                        added++;
                    }
                }
            }

            // PEV-check and evaluate
            BitSet tidSet = pevCheck(p);
            double eu = calcFitness(p, tidSet, -1);
            p.fitness = eu;

            population[i] = p;
            pBest[i] = new Particle(p.X, p.fitness);

            if (!explored.contains(p.X)) {
                if (p.fitness > minSolutionFitness) {
                    solutions.add(new Particle(p.X, p.fitness));
                }
            }

            if (i == 0) {
                gBest = new Particle(p.X, p.fitness);
            } else if (p.fitness > gBest.fitness) {
                gBest = new Particle(p.X, p.fitness);
            }

            explored.add((BitSet) p.X.clone());
        }
    }

    /**
     * Fills solution set with remaining 1-itemsets until k solutions or no more items.
     * Purpose: raise MSF quickly.
     */
    private void fillSolutions() {
        while (solutions.getSize() < k && !sizeOneItemsets.isEmpty()) {
            Item item = sizeOneItemsets.pollLast(); // highest EU 1-itemset
            if (item.totalEU <= 0) break;    // stop: remaining items have EU <= 0

            Particle p = new Particle(validItems.size());
            p.X.set(item.item);
            p.fitness = item.totalEU;
            solutions.add(p);
            explored.add(p.X);
        }
        sizeOneItemsets = null; // free memory
    }

    // =========================================================================
    // PEV-Check (Promising Encoding Vector Check)
    // =========================================================================

    /**
     * Verifies the particle exists in at least one transaction.
     * Removes items with no common transactions.
     * Also computes the per-item fitness estimate sum.
     *
     * @param p the particle to check
     * @return TID-set of the (possibly modified) particle
     */
    private BitSet pevCheck(Particle p) {
        int item = p.X.nextSetBit(0);
        if (item == -1) return new BitSet(); // empty particle

        Item firstItem = validItems.get(item - 1);
        p.estFitness = Math.abs(firstItem.avgEU); // use |avgEU| for estimation

        if (p.X.cardinality() == 1) {
            return firstItem.TIDS; // avoid clone for 1-itemsets
        }

        BitSet tidSet = (BitSet) firstItem.TIDS.clone();
        for (int i = p.X.nextSetBit(item + 1); i != -1; i = p.X.nextSetBit(i + 1)) {
            Item nextItem = validItems.get(i - 1);
            if (tidSet.intersects(nextItem.TIDS)) {
                tidSet.and(nextItem.TIDS); // intersect TID-sets
                p.estFitness += Math.abs(nextItem.avgEU);
            } else {
                p.X.clear(i); // no common transactions — remove item
            }
        }
        return tidSet;
    }

    // =========================================================================
    // Fitness Evaluation (EU computation)
    // =========================================================================

    /**
     * Computes Expected Utility of the particle's itemset.
     *
     * EU(X) = Σ_T P(X,T) × u(X,T)
     *
     * @param p      the particle
     * @param tidSet TID-set of the particle
     * @param idx    population index (-1 for initial population)
     * @return EU value, or Double.NEGATIVE_INFINITY if evaluation was skipped
     */
    private double calcFitness(Particle p, BitSet tidSet, int idx) {
        // Special case: 1-itemset — return precomputed values
        if (p.X.cardinality() == 1) {
            int itemIdx = p.X.nextSetBit(0);
            Item item = validItems.get(itemIdx - 1);
            return item.totalEU;
        }

        // Fitness estimation — skip if estimate is unpromising
        int support = tidSet.cardinality();
        if (support == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        double est = p.estFitness * support;
        double buffer = std * support;

        if (idx != -1) {
            // Skip evaluation if estimate + buffer < MSF AND estimate < pBest fitness
            if (est + buffer < minSolutionFitness && est + buffer < pBest[idx].fitness) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        // Full evaluation: compute EU in a single database scan
        double eu = 0.0;

        for (int t = tidSet.nextSetBit(0); t != -1; t = tidSet.nextSetBit(t + 1)) {
            TransEntry[] trans = database.get(t);

            // Compute u(X, T) and P(X, T) for this transaction
            double transUtil = 0.0;
            double logProb = 0.0; // log P(X, T) = Σ log P(i, T)
            int q = 0;
            int item = p.X.nextSetBit(0);

            while (item != -1 && q < trans.length) {
                if (trans[q].item == item) {
                    transUtil += trans[q].utility;
                    logProb += Math.log(trans[q].probability);
                    item = p.X.nextSetBit(item + 1);
                }
                q++;
            }

            // P(X, T) = exp(logProb)
            double prob = Math.exp(logProb);

            // EU accumulation
            eu += prob * transUtil;
        }

        // Update over/underestimate counters (for deviation tuning)
        if (est + buffer < eu) {
            lowEst++;
        } else {
            highEst++;
        }

        return eu;
    }

    // =========================================================================
    // Phase 3: PSO Update Loop
    // =========================================================================

    /**
     * Updates each particle towards pBest and gBest, then evaluates.
     */
    private void update() {
        for (int i = 0; i < pop_size; i++) {
            Particle p = population[i];

            // Update towards pBest
            List<Integer> diffList = bitDiff(pBest[i], p);
            changeParticle(diffList, p);

            // Update towards gBest
            diffList = bitDiff(gBest, p);
            changeParticle(diffList, p);

            // Random modification if particle is redundant
            if (explored.contains(p.X)) {
                int rand = (int) (validItems.size() * Math.random());
                Item item = validItems.get(rand);
                // Clear if item's PTWU < MSF
                if (item.ptwu < minSolutionFitness) {
                    p.X.clear(item.item);
                } else {
                    p.X.flip(item.item);
                }
            }

            // Evaluate if not explored
            if (!explored.contains(p.X)) {
                BitSet prePev = (BitSet) p.X.clone();
                BitSet tidSet = pevCheck(p);

                if (!explored.contains(p.X)) {
                    double eu = calcFitness(p, tidSet, i);

                    // Only process if evaluation was not skipped
                    if (eu > Double.NEGATIVE_INFINITY) {
                        p.fitness = eu;

                        // Update pBest
                        if (p.fitness > pBest[i].fitness) {
                            Particle pCopy = new Particle(p.X, p.fitness);
                            pBest[i] = pCopy;
                            if (p.fitness > gBest.fitness) {
                                gBest = pCopy;
                            }
                        }

                        // Check if top-k HUI
                        if (p.fitness > minSolutionFitness) {
                            solutions.add(new Particle(p.X, p.fitness));
                        }
                    }

                    explored.add((BitSet) p.X.clone());
                }
                explored.add(prePev); // mark pre-PEV particle as explored too
            }
        }
    }

    // =========================================================================
    // Bit Difference and Particle Modification
    // =========================================================================

    /**
     * Computes bit difference (XOR) between best and current particle.
     * @return list of differing bit positions
     */
    private List<Integer> bitDiff(Particle best, Particle p) {
        List<Integer> diffList = new ArrayList<>();
        BitSet temp = (BitSet) best.X.clone();
        temp.xor(p.X);
        for (int i = temp.nextSetBit(0); i != -1; i = temp.nextSetBit(i + 1)) {
            diffList.add(i);
        }
        return diffList;
    }

    /**
     * Flips a random number of bits from the diff list.
     * Items with PTWU < MSF are always cleared (never set).
     */
    private void changeParticle(List<Integer> diffList, Particle p) {
        if (diffList.size() > 0) {
            int num = (int) (diffList.size() * Math.random() + 1);
            for (int i = 0; i < num; i++) {
                if (diffList.isEmpty()) break;
                int pos = (int) (diffList.size() * Math.random());
                int itemIdx = diffList.remove(pos);

                // Safety check: itemIdx must be valid
                if (itemIdx < 1 || itemIdx > validItems.size()) continue;

                Item item = validItems.get(itemIdx - 1);
                // Clear if PTWU < MSF
                if (item.ptwu < minSolutionFitness) {
                    p.X.clear(item.item);
                } else {
                    p.X.flip(item.item);
                }
            }
        }
    }

    // =========================================================================
    // Roulette Wheel Selection
    // =========================================================================

    /**
     * Creates probability range for RWS on valid items, based on PTWU.
     */
    private List<Double> roulettePTWU() {
        List<Double> probRange = new ArrayList<>();
        double sum = 0;
        for (Item item : validItems) {
            sum += item.ptwu;
            probRange.add(sum / ptwuSum);
        }
        return probRange;
    }

    /**
     * Creates probability range for RWS on current top-k HUIs, based on EU.
     * Handles negative EU by shifting all values to positive.
     */
    private List<Double> rouletteTopK() {
        List<Double> probRange = new ArrayList<>();
        if (solutions.getSize() == 0) return probRange;

        // Find minimum EU to shift all values to positive
        double minEU = Double.MAX_VALUE;
        for (Particle p : solutions.getSol()) {
            minEU = Math.min(minEU, p.fitness);
        }
        double shift = (minEU < 0) ? -minEU + 1.0 : 0.0; // ensure all positive

        double totalShifted = 0;
        for (Particle p : solutions.getSol()) {
            totalShifted += (p.fitness + shift);
        }

        if (totalShifted <= 0) {
            // All equal — uniform distribution
            double step = 1.0 / solutions.getSize();
            for (int i = 0; i < solutions.getSize(); i++) {
                probRange.add((i + 1) * step);
            }
            return probRange;
        }

        double sum = 0;
        for (Particle p : solutions.getSol()) {
            sum += (p.fitness + shift);
            probRange.add(sum / totalShifted);
        }
        return probRange;
    }

    /**
     * Selects a winner from the probability range using a random number.
     */
    private int rouletteSelect(List<Double> probRange) {
        if (probRange.isEmpty()) return 0;
        double rand = Math.random();
        for (int i = 0; i < probRange.size(); i++) {
            if (rand <= probRange.get(i)) {
                return i;
            }
        }
        return probRange.size() - 1;
    }

    /**
     * Updates gBest to the top-k HUI at the given position.
     */
    private void selectGBest(int pos) {
        int c = 0;
        for (Particle p : solutions.getSol()) {
            if (c == pos) {
                gBest = p;
                break;
            }
            c++;
        }
    }

    // =========================================================================
    // Output
    // =========================================================================

    /**
     * Writes results to output file.
     */
    private void writeOutput() throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("=================================================\n");
        sb.append(String.format("UTKU-PSO Results: Top-%d Patterns%n", solutions.getSize()));
        sb.append(String.format("Parameters: k=%d, pop_size=%d, iterations=%d%n",
                k, pop_size, iterations));
        sb.append("=================================================\n");
        sb.append(String.format("%-6s %-40s %-15s%n", "Rank", "Pattern", "Expected Util"));
        sb.append("-------------------------------------------------\n");

        int rank = 1;
        for (Particle p : solutions.getSol()) {
            StringBuilder itemset = new StringBuilder("{");
            boolean first = true;
            for (int i = p.X.nextSetBit(0); i != -1; i = p.X.nextSetBit(i + 1)) {
                if (!first) itemset.append(", ");
                itemset.append(itemNamesRev.get(i));
                first = false;
            }
            itemset.append("}");

            sb.append(String.format(Locale.US, "%-6d %-40s %-15.4f%n",
                    rank++, itemset.toString(), p.fitness));
        }

        sb.append("=================================================\n");
        sb.append(String.format(Locale.US, "Execution time: %.3f seconds%n",
                (endTimestamp - startTimestamp) / 1000.0));
        sb.append(String.format("Patterns found: %d%n", solutions.getSize()));
        sb.append(String.format(Locale.US, "Memory used: %.2f MB%n", maxMemory));
        sb.append(String.format(Locale.US, "Min Solution Fitness: %.4f%n", minSolutionFitness));
        sb.append("=================================================\n");

        try (BufferedWriter w = new BufferedWriter(new FileWriter(outputFile))) {
            w.write(sb.toString());
        }

        // Also print to console
        System.out.print(sb.toString());
    }

    /**
     * Prints execution statistics.
     */
    public void printStats() {
        System.out.println("============= UTKU-PSO STATS ==============");
        System.out.println(" Total time ~ " + (endTimestamp - startTimestamp) + " ms");
        System.out.printf(Locale.US, " Memory ~ %.2f MB%n", maxMemory);
        System.out.printf(Locale.US, " Sum EU of top-k: %.4f%n", euSum);
        System.out.printf(Locale.US, " Min Solution Fitness: %.4f%n", minSolutionFitness);
        System.out.println(" Solutions found: " + solutions.getSize());
        System.out.println(" Valid items: " + validItems.size());
        System.out.println(" Fitness evaluations skipped (estimate): " + highEst);
        System.out.println(" Fitness underestimates: " + lowEst);
        System.out.println("============================================");
    }

    private void checkMemory() {
        double currentMemory = (Runtime.getRuntime().totalMemory() - Runtime
                .getRuntime().freeMemory()) / 1024d / 1024d;
        maxMemory = Math.max(maxMemory, currentMemory);
    }
}