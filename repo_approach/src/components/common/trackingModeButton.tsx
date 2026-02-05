import { useDispatch } from "react-redux";
import { settingsSelect, toggleTrackingMode } from "../../slices/settingsSlice";
import SidebarButton from "./sidebarButton";

const TrackingModeButton = () => {
  const dispatch = useDispatch();
  const settings = settingsSelect();

  const handleClick = () => {
    dispatch(toggleTrackingMode());
  };

  return (
    <SidebarButton onClick={handleClick}>
      {settings.usePositionTracking ? "🎯 Position Tracking" : "📊 Score-Based"}
    </SidebarButton>
  );
};

export default TrackingModeButton;
