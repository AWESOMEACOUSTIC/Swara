"use client";

import { create } from "zustand";

export type PlayerTrack = {
  id: string;
  title: string | null;
  url: string;
  artwork: string | null;
  prompt: string | null;
  createdByUserName: string | null;
};

type PlayerState = {
  track: PlayerTrack | null;
  setTrack: (track: PlayerTrack) => void;
};

export const usePlayerStore = create<PlayerState>((set) => ({
  track: null,
  setTrack: (track) => set({ track }),
}));
