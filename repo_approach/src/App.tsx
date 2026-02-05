import { useRef } from "react";
import { Outlet } from "react-router-dom";
import { GraphModel } from "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import { ModelRefs } from "./types";

const App = () => {
  const piecesModelRef = useRef<GraphModel>();
  const xcornersModelRef = useRef<GraphModel>();
  const modelRefs: ModelRefs = {
    "piecesModelRef": piecesModelRef,
    "xcornersModelRef": xcornersModelRef,           
  }

  return (
    <>
      <Outlet context={modelRefs}/>
    </>
  );
};

export default App;