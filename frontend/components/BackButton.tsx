'use client';

import { useRouter } from 'next/navigation';

export default function BackButton({ className = '' }: { className?: string }) {
  const router = useRouter();
  return (
    <button
      onClick={() => router.back()}
      className={`px-4 py-2 rounded-full text-sm transition btn-neutral ${className}`}
    >
      뒤로가기
    </button>
  );
}


