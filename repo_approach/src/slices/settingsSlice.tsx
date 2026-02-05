import { createSlice } from "@reduxjs/toolkit";
import { useSelector } from "react-redux";

interface SettingsState {
  usePositionTracking: boolean;
}

const initialState: SettingsState = {
  usePositionTracking: true, // Default to our new approach
};

const settingsSlice = createSlice({
  name: "settings",
  initialState,
  reducers: {
    toggleTrackingMode: (state) => {
      state.usePositionTracking = !state.usePositionTracking;
    },
    setTrackingMode: (state, action) => {
      state.usePositionTracking = action.payload;
    },
  },
});

export const { toggleTrackingMode, setTrackingMode } = settingsSlice.actions;

export const settingsSelect = (): SettingsState => {
  return useSelector((state: any) => state.settings);
};

export default settingsSlice.reducer;
