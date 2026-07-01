import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "RAG Assistant",
  description: "Document Q&A powered by hybrid retrieval and a local LLM",
  applicationName: "RAG Assistant",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "RAG Assistant",
    statusBarStyle: "black-translucent",
  },
  icons: {
    icon: "/icon-192.png",
    apple: "/apple-touch-icon.png",
  },
};

export const viewport: Viewport = {
  themeColor: "#07070b",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
