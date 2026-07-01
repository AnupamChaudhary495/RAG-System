import type { SVGProps } from "react";

type Icon = (props: SVGProps<SVGSVGElement>) => JSX.Element;

const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  viewBox: "0 0 24 24",
};

export const SparkIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
    <path d="M12 8.5 13.4 11 16 12l-2.6 1-1.4 2.5L10.6 13 8 12l2.6-1L12 8.5Z" />
  </svg>
);

export const SendIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </svg>
);

export const DocIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
    <path d="M14 3v5h5M9 13h6M9 17h4" />
  </svg>
);

export const ChevronIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="m6 9 6 6 6-6" />
  </svg>
);

export const PlusIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const UserIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M20 21a8 8 0 0 0-16 0" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

export const CheckIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M20 6 9 17l-5-5" />
  </svg>
);

export const MenuIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M4 6h16M4 12h16M4 18h16" />
  </svg>
);

export const SidebarIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </svg>
);

export const TrashIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M4 7h16M10 11v6M14 11v6M5 7l1 13a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-13M9 7V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3" />
  </svg>
);

export const EditIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
  </svg>
);

export const StopIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" stroke="none" />
  </svg>
);

export const CopyIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h8" />
  </svg>
);

export const ArrowDownIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 5v14M5 12l7 7 7-7" />
  </svg>
);

export const ChatIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M21 15a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z" />
  </svg>
);

export const DownloadIcon: Icon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3v12M7 10l5 5 5-5M5 21h14" />
  </svg>
);
