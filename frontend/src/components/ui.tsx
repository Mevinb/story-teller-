import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";

export function ErrorNotice({
  error,
  retry,
}: {
  error?: string;
  retry?: () => void;
}) {
  return error ? (
    <div className="notice error" role="alert">
      {error}
      {retry && <button onClick={retry}>Try again</button>}
    </div>
  ) : null;
}
export function Loading() {
  return (
    <div className="loading" role="status" aria-label="Loading">
      <span />
      <span />
      <span />
    </div>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <span className="empty-mark" aria-hidden="true">
        S.
      </span>
      <h2>{title}</h2>
      {children}
    </div>
  );
}
export function PageTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <header className="page-title">
      <div>
        <p className="eyebrow">{eyebrow || "Writing workspace"}</p>
        <h1>{title}</h1>
      </div>
      {children}
    </header>
  );
}
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
export function Prose({ text }: { text: string }) {
  return (
    <div className="prose">
      {text
        .split(/\n\n+/)
        .map((p, i) =>
          /^#{1,3}\s/.test(p) ? (
            <h2 key={i}>{p.replace(/^#+\s*/, "")}</h2>
          ) : (
            <p key={i}>{p}</p>
          ),
        )}
    </div>
  );
}
export function JsonView({ value }: { value: unknown }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}
export function Dialog({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const id = useId();
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const dialog = ref.current;
    dialog?.showModal();
    return () => {
      dialog?.close();
      previous?.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      aria-labelledby={id}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <div className="dialog-heading">
        <h2 id={id}>{title}</h2>
        <button aria-label="Close dialog" onClick={onClose}>
          ×
        </button>
      </div>
      {children}
    </dialog>
  );
}
interface UiContextValue {
  notify: (text: string) => void;
  confirm: (text: string) => Promise<boolean>;
}
const UiContext = createContext<UiContextValue>(null!);
export const useUi = () => useContext(UiContext);
export function UiProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState("");
  const [confirmation, setConfirmation] = useState<{
    text: string;
    resolve: (answer: boolean) => void;
  }>();
  const notify = useCallback((text: string) => setToast(text), []);
  const confirm = useCallback(
    (text: string) =>
      new Promise<boolean>((resolve) => setConfirmation({ text, resolve })),
    [],
  );
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 4500);
      return () => clearTimeout(timer);
    }
  }, [toast]);
  const answer = (value: boolean) => {
    confirmation?.resolve(value);
    setConfirmation(undefined);
  };
  return (
    <UiContext.Provider value={{ notify, confirm }}>
      {children}
      <div className="toast" role="status">
        {toast}
      </div>
      {confirmation && (
        <Dialog title="Confirm action" onClose={() => answer(false)}>
          <p>{confirmation.text}</p>
          <div className="actions">
            <button autoFocus onClick={() => answer(false)}>
              Keep working
            </button>
            <button className="danger" onClick={() => answer(true)}>
              Confirm
            </button>
          </div>
        </Dialog>
      )}
    </UiContext.Provider>
  );
}
