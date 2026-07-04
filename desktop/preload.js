// Safe bridge between renderer and main. No node in the renderer; just these.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("ghost", {
  backendUrl: () => ipcRenderer.invoke("backend-url"),
  pickBinary: () => ipcRenderer.invoke("pick-binary"),
  onBackendReady: (cb) => ipcRenderer.on("backend-ready", cb),
});
