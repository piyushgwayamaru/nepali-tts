import React, { useRef, useState } from "react";

type AudioFormat = "mp3" | "wav" | "ogg";

export default function App() {
  const [text, setText] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [format, setFormat] = useState<AudioFormat>("wav");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleSynthesis = async () => {
    if (!text.trim()) return alert("Please enter some text.");
    setIsLoading(true);
    try {
      const response = await fetch(
        `http://localhost:9000/synthesize?text=${encodeURIComponent(text)}`
      );
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioUrl(audioUrl);
    } catch (error) {
      alert("Error during synthesis.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!audioUrl) {
      alert("Please generate audio first by clicking Speak");
      return;
    }

    const link = document.createElement("a");
    link.href = audioUrl;
    link.download = `nepali-tts.${format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Nepali Text to Speech
          </h1>
          <p className="mt-3 text-xl text-gray-500">
            Convert your Nepali text to speech
          </p>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <div className="mb-6">
            <label
              htmlFor="text"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Nepali Text
            </label>
            <textarea
              id="text"
              rows={4}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Type Nepali text here..."
              value={text}
              onChange={(e) => {
                const input = e.target.value;
                const nepaliOnlyRegex = /^[\u0900-\u097F\s।,!?\-–—।\n\r\t]+$/;

                if (input === "" || nepaliOnlyRegex.test(input)) {
                  setText(input);
                  setError("");
                } else {
                  setError(
                    "Only Nepali characters and punctuation are allowed. English letters or numbers are not allowed."
                  );
                }
              }}
            />

            {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
          </div>
        </div>

        {/* ✅ Extra spacing between textarea and buttons */}
        <div className="mt-6 flex flex-wrap justify-center gap-4 items-center">
          <button
            type="button"
            onClick={handleSynthesis}
            disabled={isLoading}
            className={`flex items-center gap-2 bg-black text-white rounded-md px-4 py-2 ${
              isLoading ? "opacity-70 cursor-not-allowed" : ""
            }`}
          >
            {isLoading && (
              <svg
                className="animate-spin h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                />
              </svg>
            )}
            {isLoading ? "Synthesizing..." : "Speak"}
          </button>

          <button
            type="button"
            onClick={handleDownload}
            className="inline-flex items-center px-6 py-3 border border-gray-300 shadow-sm text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <svg
              className="w-5 h-5 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            Download
          </button>
        </div>

        {audioUrl && (
          <div className="mt-8">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Audio Player
            </label>
            <audio ref={audioRef} src={audioUrl} controls className="w-full">
              Your browser does not support the audio element.
            </audio>
          </div>
        )}
      </div>
    </div>
  );
}
