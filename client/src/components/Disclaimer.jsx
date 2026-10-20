import { DISCLAIMER } from "../constants.js";

export default function Disclaimer({ text = DISCLAIMER }) {
  return (
    <div role="note" className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
      <strong className="font-semibold">Not legal advice. </strong>
      {text}
    </div>
  );
}
