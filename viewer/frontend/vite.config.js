import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  base: "/viewer/",

  build: {
    target: "esnext",

    outDir: resolve(
      process.cwd(),
      "../../src/fastgc/viewer/web",
    ),

    emptyOutDir: true,

    rollupOptions: {
      input: resolve(
        process.cwd(),
        "index.html",
      ),
    },
  },

  optimizeDeps: {
    esbuildOptions: {
      target: "esnext",
    },
  },
});
