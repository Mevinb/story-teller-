export type View =
  | "library"
  | "create"
  | "edit"
  | "premise"
  | "generate"
  | "manual"
  | "reader"
  | "combine"
  | "bible"
  | "logs"
  | "vision"
  | "settings";
export interface Character {
  description?: string;
  traits?: string[];
  [key: string]: unknown;
}
export interface Metadata {
  title?: string;
  genre?: string;
  premise?: string;
  setting?: string;
  themes?: string[];
  current_chapter?: number;
  [key: string]: unknown;
}
export interface StoryState {
  metadata: Metadata;
  characters: Record<string, Character>;
  [key: string]: unknown;
}
export interface Project {
  name: string;
  title: string;
  genre: string;
  premise: string;
  total_chapters: number;
  word_count: number;
  characters: string[];
  created_at?: string;
}
export interface Model {
  id: string;
  name?: string;
  description?: string;
  context_window?: number;
  [key: string]: unknown;
}
export interface Catalog {
  active: string;
  backend_mode: string;
  groq_models: Model[];
  gemini_models: Model[];
  openrouter_models: Model[];
  local_models: string[];
  gemini_model: string;
}
export interface Chapter {
  number: number;
  title: string;
  words: number;
  status: string;
}
export interface ChapterText {
  chapter: number;
  content: string;
  title: string;
  status?: string;
}
export interface ManualStatus {
  active: boolean;
  chapter_num?: number;
  chapter_title?: string;
  scenes_completed?: number;
  completed_scenes?: (string | { text: string })[];
  is_generating?: boolean;
}
export interface Version {
  suffix: string;
  label: string;
  timestamp: number;
}
export interface PromptHighlight {
  title: string;
  context: string;
  prompt: string;
  references: string[];
}
export interface PromptResult {
  status: string;
  highlights: PromptHighlight[];
  error?: string;
}
export type SettingValue = string | number | boolean;
export interface Settings {
  settings: Record<string, SettingValue>;
}
export interface StreamEvent {
  type: string;
  timestamp?: string;
  content?: string;
  payload?: string | Record<string, unknown>;
  data?: string | Record<string, unknown>;
}
export interface Activity {
  generation: boolean;
  combine: boolean;
  manual: boolean;
}
