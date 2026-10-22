
import Link from "next/link";
import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-white min-h-screen">
        <nav className="border-b border-slate-800 p-5 flex gap-8">
          <Link href="/" className="font-bold text-green-400">
            FPL Predictor
          </Link>
          <Link href="/players">Players</Link>
          <Link href="/team">Best XI</Link>
          <Link href="/fixtures">Fixtures</Link>
        </nav>

        <main className="max-w-6xl mx-auto p-8">
          {children}
        </main>
      </body>
    </html>
  );
}
