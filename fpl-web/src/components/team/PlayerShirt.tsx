
import { teamColors } from "@/lib/teamColors";

type Props = {
  team: string;
};

export default function PlayerShirt({ team }: Props) {
  const color = teamColors[team] || "#a3e635";

  return (
    <svg
      viewBox="0 0 80 80"
      className="w-12 h-12 sm:w-16 sm:h-16 drop-shadow-lg"
    >
      <path
        d="M24 10 L32 6 L48 6 L56 10 L74 22 L65 39 L57 34 L57 72 L23 72 L23 34 L15 39 L6 22 Z"
        fill={color}
        stroke="white"
        strokeWidth="2"
      />
      <path
        d="M32 6 Q40 21 48 6"
        fill="none"
        stroke="white"
        strokeWidth="3"
      />
    </svg>
  );
}
