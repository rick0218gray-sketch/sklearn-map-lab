import React from "react";
import { createRoot } from "react-dom/client";
import AutoAnalysis from "./AutoAnalysis.jsx";
import "./page.css";

createRoot(document.getElementById("root")).render(
  <main>
    <header>
      <h1>sklearn-map-lab</h1>
      <p>候補モデルを同じ条件で比較し、採用理由と検証結果を残す</p>
    </header>
    <AutoAnalysis />
  </main>
);
