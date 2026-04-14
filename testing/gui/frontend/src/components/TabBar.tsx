import { useStore } from "../state/store";
import type { ConfigTab } from "../types/api";

const TABS: { id: ConfigTab; label: string }[] = [
  { id: "sessions", label: "Sessions" },
  { id: "probe", label: "Probe" },
  { id: "intervene", label: "Intervene" },
];

export function TabBar() {
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

  return (
    <div className="tab-bar">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
          onClick={() => setActiveTab(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
