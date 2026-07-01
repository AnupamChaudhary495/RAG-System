import type { ButtonHTMLAttributes } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement>;

/**
 * A button with an animated rotating gradient "beam" border — a polished,
 * classy micro-interaction in the spirit of open-source component libraries
 * (Magic UI / Aceternity). Pure CSS; see `.shimmer-btn` in globals.css.
 */
export function ShimmerButton({ children, className = "", ...props }: Props) {
  return (
    <button className={`shimmer-btn ${className}`} {...props}>
      <span className="shimmer-btn__content">{children}</span>
    </button>
  );
}
