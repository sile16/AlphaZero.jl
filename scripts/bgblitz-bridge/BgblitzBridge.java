/**
 * BgblitzBridge - Headless bridge to BGBlitz's TachiAI position evaluator.
 *
 * Loads BGBlitz's neural network evaluator without the GUI and communicates
 * via stdin/stdout for integration with AlphaZero.jl evaluation.
 *
 * Protocol:
 *   Input:  BESTMOVE XGID=...    (get best move for position with dice)
 *   Output: MOVE from1/to1,from2/to2|equity pWin pGamWin pBGWin pGamLoss pBGLoss
 *
 *   Input:  EVAL XGID=...        (evaluate position)
 *   Output: EVAL equity pWin pGamWin pBGWin pGamLoss pBGLoss
 *
 *   Input:  QUIT
 *   (process exits)
 *
 * Compile:
 *   BGBLITZ_APP=/Applications/BGBlitz/Bgblitz.app/Contents/app
 *   JAVA=$BGBLITZ_APP/../runtime/Contents/Home/bin/javac
 *   $JAVA --enable-preview --add-modules jdk.incubator.vector \
 *     -cp "$BGBLITZ_APP/bgblitz.jar:$BGBLITZ_APP/tools.jar:$BGBLITZ_APP/res.jar" \
 *     -d scripts/bgblitz-bridge scripts/bgblitz-bridge/BgblitzBridge.java
 *
 * Run:
 *   JAVA=$BGBLITZ_APP/../runtime/Contents/Home/bin/java
 *   $JAVA -Xmx512m -DBGB_SYSTEM_DIR=/Applications/BGBlitz/system \
 *     --enable-preview --add-modules jdk.incubator.vector \
 *     -cp "$BGBLITZ_APP/bgblitz.jar:$BGBLITZ_APP/tools.jar:$BGBLITZ_APP/res.jar:$BGBLITZ_APP/tools_jni.jar:scripts/bgblitz-bridge" \
 *     BgblitzBridge [ply]
 */

import bgblitz.XGID;
import bgblitz.Position;
import bgblitz.Move;
import bgblitz.bot.Equity;
import bgblitz.play.TachiAIPosEvaluator;
import bgblitz.play.MoveEquityPair;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.List;

public class BgblitzBridge {

    private static TachiAIPosEvaluator evaluator;
    private static int searchPly = 0;
    // Protocol output stream (saved before redirecting System.out)
    private static PrintStream protocolOut;

    public static void main(String[] args) {
        // Save original stdout for our protocol, then redirect System.out to stderr.
        // This prevents BGBlitz's internal logging (TachiAI search debug, etc.)
        // from contaminating our stdin/stdout protocol.
        protocolOut = System.out;
        System.setOut(System.err);

        // Parse ply from args
        if (args.length > 0) {
            try {
                searchPly = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Invalid ply argument: " + args[0]);
                searchPly = 0;
            }
        }

        // Initialize evaluator
        try {
            initEvaluator();
        } catch (Exception e) {
            protocolOut.println("ERROR Failed to initialize evaluator: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }

        protocolOut.println("READY ply=" + searchPly);
        protocolOut.flush();

        // Main command loop
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (line.equals("QUIT")) break;

                try {
                    if (line.startsWith("BESTMOVE ")) {
                        handleBestMove(line.substring(9).trim());
                    } else if (line.startsWith("EVAL ")) {
                        handleEval(line.substring(5).trim());
                    } else {
                        protocolOut.println("ERROR Unknown command: " + line);
                    }
                } catch (Exception e) {
                    protocolOut.println("ERROR " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    System.err.println("Error processing: " + line);
                    e.printStackTrace(System.err);
                }
                protocolOut.flush();
            }
        } catch (Exception e) {
            System.err.println("Fatal error in command loop: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    private static void initEvaluator() throws Exception {
        System.err.println("Initializing BGBlitz evaluator (ply=" + searchPly + ")...");

        // Try factory method first (reads config from BGB_SYSTEM_DIR)
        try {
            evaluator = bgblitz.bot.tachiai.p.e();
            if (evaluator != null) {
                evaluator.setSearchDepth(searchPly);
                System.err.println("Evaluator initialized via factory: " + evaluator.evaluatorInfo());
                return;
            }
        } catch (Exception e) {
            System.err.println("Factory method p.e() failed: " + e.getMessage());
        }

        // Try alternate factory
        try {
            evaluator = bgblitz.bot.tachiai.p.c();
            if (evaluator != null) {
                evaluator.setSearchDepth(searchPly);
                System.err.println("Evaluator initialized via factory c(): " + evaluator.evaluatorInfo());
                return;
            }
        } catch (Exception e) {
            System.err.println("Factory method p.c() failed: " + e.getMessage());
        }

        // Direct instantiation fallback: TachiAI5_nf with v5_192_nf nets
        try {
            String systemDir = System.getProperty("BGB_SYSTEM_DIR", "/Applications/BGBlitz/system");
            String netPath = systemDir + "/bot/tachiai/v5_192_nf";
            System.err.println("Trying direct instantiation with net path: " + netPath);

            bgblitz.bot.tachiai.TachiAI5_nf tachiai = new bgblitz.bot.tachiai.TachiAI5_nf(netPath);
            evaluator = new TachiAIPosEvaluator(tachiai);
            evaluator.setSearchDepth(searchPly);
            System.err.println("Evaluator initialized via direct instantiation");
            return;
        } catch (Exception e) {
            System.err.println("Direct TachiAI5_nf instantiation failed: " + e.getMessage());
        }

        // Try with just the net name (no path)
        try {
            bgblitz.bot.tachiai.TachiAI5_nf tachiai = new bgblitz.bot.tachiai.TachiAI5_nf("v5_192_nf");
            evaluator = new TachiAIPosEvaluator(tachiai);
            evaluator.setSearchDepth(searchPly);
            System.err.println("Evaluator initialized via direct instantiation (name only)");
            return;
        } catch (Exception e) {
            System.err.println("Direct TachiAI5_nf (name only) instantiation failed: " + e.getMessage());
        }

        // Try default constructor
        try {
            evaluator = new TachiAIPosEvaluator();
            evaluator.setSearchDepth(searchPly);
            System.err.println("Evaluator initialized via default constructor");
            return;
        } catch (Exception e) {
            System.err.println("Default constructor failed: " + e.getMessage());
        }

        throw new RuntimeException("All evaluator initialization methods failed");
    }

    private static void handleBestMove(String xgidStr) {
        XGID xgid = new XGID(xgidStr);
        Position pos = new Position(xgid);

        List<MoveEquityPair> moves = evaluator.getBestMoves(pos);

        if (moves == null || moves.isEmpty()) {
            protocolOut.println("MOVE PASS|0.000000 0.500000 0.000000 0.000000 0.000000 0.000000");
            return;
        }

        MoveEquityPair best = moves.get(0);
        Move move = best.move;
        Equity eq = best.eq;

        StringBuilder sb = new StringBuilder("MOVE ");

        // Format move: from/to pairs
        int nMoves = move.noOfElementaryMoves();
        if (nMoves == 0 || move.isDancing()) {
            sb.append("PASS");
        } else {
            for (int i = 0; i < nMoves; i++) {
                if (i > 0) sb.append(",");
                sb.append(move.from[i]).append("/").append(move.to[i]);
            }
        }

        sb.append("|");
        appendEquity(sb, eq);

        protocolOut.println(sb.toString());
    }

    private static void handleEval(String xgidStr) {
        XGID xgid = new XGID(xgidStr);
        Position pos = new Position(xgid);

        Equity eq = evaluator.evalPosition(pos, true);

        StringBuilder sb = new StringBuilder("EVAL ");
        appendEquity(sb, eq);

        protocolOut.println(sb.toString());
    }

    private static void appendEquity(StringBuilder sb, Equity eq) {
        // Output: equity pWin pGamWin pBGWin pGamLoss pBGLoss
        // "true" = from on-roll player's perspective
        sb.append(String.format("%.6f %.6f %.6f %.6f %.6f %.6f",
            eq.getEquity(true),
            eq.getWins(true),
            eq.getGammon(true),
            eq.getBackGammon(true),
            eq.getGammon(false),
            eq.getBackGammon(false)));
    }
}
