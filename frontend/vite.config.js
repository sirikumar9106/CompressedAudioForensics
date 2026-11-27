import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// `npm run build` writes one self-contained index.html (code, styles and fonts inlined)
// straight into the folder FastAPI serves, so the app works offline with no extra routes.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: { outDir: "../src/app/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:7860" } },
});
