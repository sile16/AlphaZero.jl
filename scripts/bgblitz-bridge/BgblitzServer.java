/**
 * BgblitzServer - TCP server wrapping BGBlitz's TachiAI evaluator.
 *
 * Single JVM with N evaluator instances, served over TCP. Avoids spawning
 * multiple JVMs and prevents macOS focus-stealing via headless mode.
 *
 * Protocol (per TCP connection):
 *   Server -> Client: READY
 *   Client -> Server: BESTMOVE XGID=...
 *   Server -> Client: MOVE from/to,...|equity pWin pGamWin pBGWin pGamLoss pBGLoss
 *   Client -> Server: EVAL XGID=...
 *   Server -> Client: EVAL equity pWin pGamWin pBGWin pGamLoss pBGLoss
 *   Client -> Server: QUIT  (closes connection, returns evaluator to pool)
 *
 * Lifecycle:
 *   Prints "READY port=N" on stdout when ready for connections.
 *   Shuts down when QUIT is received on stdin.
 *
 * Compile:
 *   BGBLITZ_APP=/Applications/BGBlitz/Bgblitz.app/Contents/app
 *   JAVA=$BGBLITZ_APP/../runtime/Contents/Home/bin/javac
 *   $JAVA --enable-preview --add-modules jdk.incubator.vector \
 *     -cp "$BGBLITZ_APP/bgblitz.jar:$BGBLITZ_APP/tools.jar:$BGBLITZ_APP/res.jar" \
 *     -d scripts/bgblitz-bridge scripts/bgblitz-bridge/BgblitzServer.java
 *
 * Run:
 *   JAVA=$BGBLITZ_APP/../runtime/Contents/Home/bin/java
 *   $JAVA -Djava.awt.headless=true -Dapple.awt.UIElement=true \
 *     -Xmx512m -DBGB_SYSTEM_DIR=/Applications/BGBlitz/system \
 *     --enable-preview --add-modules jdk.incubator.vector \
 *     -cp "$BGBLITZ_APP/bgblitz.jar:$BGBLITZ_APP/tools.jar:$BGBLITZ_APP/res.jar:$BGBLITZ_APP/tools_jni.jar:scripts/bgblitz-bridge" \
 *     BgblitzServer --slots 4 --ply 0
 */

import bgblitz.XGID;
import bgblitz.Position;
import bgblitz.Move;
import bgblitz.bot.Equity;
import bgblitz.play.TachiAIPosEvaluator;
import bgblitz.play.MoveEquityPair;

import java.io.*;
import java.net.*;
import java.util.List;
import java.util.concurrent.*;

public class BgblitzServer {

    private static PrintStream protocolOut;

    public static void main(String[] args) {
        // Save stdout for protocol, redirect System.out to stderr
        protocolOut = System.out;
        System.setOut(System.err);

        int ply = 0;
        int slots = 4;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ply": ply = Integer.parseInt(args[++i]); break;
                case "--slots": slots = Integer.parseInt(args[++i]); break;
            }
        }

        System.err.println("BgblitzServer: initializing " + slots + " evaluators (ply=" + ply + ")...");

        // Create evaluator pool
        BlockingQueue<TachiAIPosEvaluator> pool = new ArrayBlockingQueue<>(slots);
        for (int i = 0; i < slots; i++) {
            try {
                TachiAIPosEvaluator eval = initEvaluator(ply);
                pool.put(eval);
                System.err.println("  Evaluator " + (i + 1) + "/" + slots + " ready");
            } catch (Exception e) {
                protocolOut.println("ERROR Failed to init evaluator " + i + ": " + e.getMessage());
                protocolOut.flush();
                System.exit(1);
            }
        }

        // Start TCP server on auto-assigned port
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            serverSocket.setReuseAddress(true);
            int port = serverSocket.getLocalPort();

            protocolOut.println("READY port=" + port);
            protocolOut.flush();
            System.err.println("Server listening on port " + port);

            // Accept connections in background thread
            Thread acceptThread = new Thread(() -> {
                while (!serverSocket.isClosed()) {
                    try {
                        Socket client = serverSocket.accept();
                        client.setTcpNoDelay(true); // minimize latency
                        System.err.println("Client connected: " + client.getRemoteSocketAddress());
                        new Thread(() -> handleClient(client, pool), "client-handler").start();
                    } catch (SocketException e) {
                        break; // server socket closed
                    } catch (Exception e) {
                        System.err.println("Accept error: " + e.getMessage());
                    }
                }
            }, "accept-thread");
            acceptThread.setDaemon(true);
            acceptThread.start();

            // Wait for QUIT on stdin
            BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
            String line;
            while ((line = stdin.readLine()) != null) {
                if (line.trim().equals("QUIT")) break;
            }
            System.err.println("Shutting down...");
        } catch (Exception e) {
            protocolOut.println("ERROR Server failed: " + e.getMessage());
            protocolOut.flush();
            System.exit(1);
        }
    }

    private static void handleClient(Socket client, BlockingQueue<TachiAIPosEvaluator> pool) {
        TachiAIPosEvaluator evaluator = null;
        try {
            evaluator = pool.take(); // borrow from pool (blocks if none available)

            BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
            PrintStream out = new PrintStream(client.getOutputStream(), true);

            out.println("READY");

            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (line.equals("QUIT")) break;

                try {
                    if (line.startsWith("BESTMOVE ")) {
                        handleBestMove(line.substring(9).trim(), evaluator, out);
                    } else if (line.startsWith("EVAL ")) {
                        handleEval(line.substring(5).trim(), evaluator, out);
                    } else {
                        out.println("ERROR Unknown command: " + line);
                    }
                } catch (Exception e) {
                    out.println("ERROR " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    System.err.println("Error processing: " + line);
                    e.printStackTrace(System.err);
                }
                out.flush();
            }
        } catch (Exception e) {
            System.err.println("Client handler error: " + e.getMessage());
        } finally {
            if (evaluator != null) {
                try { pool.put(evaluator); } catch (InterruptedException e) {}
            }
            try { client.close(); } catch (Exception e) {}
            System.err.println("Client disconnected, evaluator returned to pool");
        }
    }

    // =========================================================================
    // Evaluator initialization (same as BgblitzBridge)
    // =========================================================================

    private static TachiAIPosEvaluator initEvaluator(int ply) throws Exception {
        // Try factory method first
        try {
            TachiAIPosEvaluator eval = bgblitz.bot.tachiai.p.e();
            if (eval != null) {
                eval.setSearchDepth(ply);
                return eval;
            }
        } catch (Exception e) {
            System.err.println("  Factory p.e() failed: " + e.getMessage());
        }

        // Try alternate factory
        try {
            TachiAIPosEvaluator eval = bgblitz.bot.tachiai.p.c();
            if (eval != null) {
                eval.setSearchDepth(ply);
                return eval;
            }
        } catch (Exception e) {
            System.err.println("  Factory p.c() failed: " + e.getMessage());
        }

        // Direct instantiation with path
        try {
            String systemDir = System.getProperty("BGB_SYSTEM_DIR", "/Applications/BGBlitz/system");
            String netPath = systemDir + "/bot/tachiai/v5_192_nf";
            bgblitz.bot.tachiai.TachiAI5_nf tachiai = new bgblitz.bot.tachiai.TachiAI5_nf(netPath);
            TachiAIPosEvaluator eval = new TachiAIPosEvaluator(tachiai);
            eval.setSearchDepth(ply);
            return eval;
        } catch (Exception e) {
            System.err.println("  Direct instantiation failed: " + e.getMessage());
        }

        // Direct instantiation with name only
        try {
            bgblitz.bot.tachiai.TachiAI5_nf tachiai = new bgblitz.bot.tachiai.TachiAI5_nf("v5_192_nf");
            TachiAIPosEvaluator eval = new TachiAIPosEvaluator(tachiai);
            eval.setSearchDepth(ply);
            return eval;
        } catch (Exception e) {
            System.err.println("  Direct instantiation (name only) failed: " + e.getMessage());
        }

        // Default constructor
        try {
            TachiAIPosEvaluator eval = new TachiAIPosEvaluator();
            eval.setSearchDepth(ply);
            return eval;
        } catch (Exception e) {
            System.err.println("  Default constructor failed: " + e.getMessage());
        }

        throw new RuntimeException("All evaluator initialization methods failed");
    }

    // =========================================================================
    // Command handlers
    // =========================================================================

    private static void handleBestMove(String xgidStr, TachiAIPosEvaluator evaluator, PrintStream out) {
        XGID xgid = new XGID(xgidStr);
        Position pos = new Position(xgid);

        List<MoveEquityPair> moves = evaluator.getBestMoves(pos);

        if (moves == null || moves.isEmpty()) {
            out.println("MOVE PASS|0.000000 0.500000 0.000000 0.000000 0.000000 0.000000");
            return;
        }

        MoveEquityPair best = moves.get(0);
        Move move = best.move;
        Equity eq = best.eq;

        StringBuilder sb = new StringBuilder("MOVE ");

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
        out.println(sb.toString());
    }

    private static void handleEval(String xgidStr, TachiAIPosEvaluator evaluator, PrintStream out) {
        XGID xgid = new XGID(xgidStr);
        Position pos = new Position(xgid);

        Equity eq = evaluator.evalPosition(pos, true);

        StringBuilder sb = new StringBuilder("EVAL ");
        appendEquity(sb, eq);
        out.println(sb.toString());
    }

    private static void appendEquity(StringBuilder sb, Equity eq) {
        sb.append(String.format("%.6f %.6f %.6f %.6f %.6f %.6f",
            eq.getEquity(true),
            eq.getWins(true),
            eq.getGammon(true),
            eq.getBackGammon(true),
            eq.getGammon(false),
            eq.getBackGammon(false)));
    }
}
