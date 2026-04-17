// Minimal IndexedDB adapter conforming to zustand's StateStorage interface.
//
// Rationale for IDB over localStorage: persisted state can include dozens of
// streaming probe results (logit-lens predictions + optional hidden-state
// blobs), which routinely exceeds localStorage's ~5MB cap on Chromium and
// corrupts the save. IDB gives us a much larger quota (tens to hundreds of
// MB, configurable by the browser) and async I/O that won't jank the main
// thread when the store hits persist-midnight.
//
// We still keep the payload small by partialize-stripping hidden-state b64
// blobs and capping result count in the store itself — this adapter is the
// storage floor, not a license to persist everything.

import type { StateStorage } from "zustand/middleware";

const DB_NAME = "llm-surgeon-gui";
const DB_VERSION = 1;
const STORE_NAME = "kv";

// One open-per-call keeps the adapter stateless and avoids the "db handle
// went stale after a long idle period" class of bug. IDB open is fast
// (<1ms after first run); persistence hits are rare enough that the
// overhead is invisible.
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error("IndexedDB upgrade blocked"));
  });
}

function txOp<T>(
  mode: IDBTransactionMode,
  op: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, mode);
        const store = tx.objectStore(STORE_NAME);
        const req = op(store);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

export const idbStorage: StateStorage = {
  getItem: async (key) => {
    try {
      const v = await txOp<unknown>("readonly", (s) => s.get(key));
      return typeof v === "string" ? v : v == null ? null : String(v);
    } catch {
      return null;
    }
  },
  setItem: async (key, value) => {
    try {
      await txOp<IDBValidKey>("readwrite", (s) => s.put(value, key));
    } catch {
      // Quota or access errors: drop silently. Losing persistence is
      // preferable to throwing and breaking the user's current session.
    }
  },
  removeItem: async (key) => {
    try {
      await txOp<undefined>("readwrite", (s) => s.delete(key));
    } catch {
      /* see setItem */
    }
  },
};
