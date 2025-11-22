'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { likesAPI } from '@/lib/api';

type Props = {
  movieId: string;
  initiallyLiked?: boolean;
};

export default function MovieActions({ movieId, initiallyLiked = false }: Props) {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [liked, setLiked] = useState<boolean>(initiallyLiked);
  const [pending, setPending] = useState<boolean>(false);

  // Sync state if prop changes, and also double-check with API if authenticated
  useEffect(() => {
    setLiked(initiallyLiked);

    if (isAuthenticated && movieId) {
      // Double check status to be sure
      likesAPI.check([{ type: 'Movie', id: movieId }])
          .then(res => {
            const isLiked = res.data?.liked?.[`Movie_${movieId}`];
            if (typeof isLiked === 'boolean') {
              setLiked(isLiked);
            }
          })
          .catch(() => {});
    }
  }, [initiallyLiked, movieId, isAuthenticated]);

  const handleBack = () => {
    window.history.back();
  };

  const toggleLike = async () => {
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    if (pending) return;
    setPending(true);

    // Optimistic Update
    const previousState = liked;
    setLiked(!previousState);

    try {
      if (!previousState) {
        await likesAPI.create('Movie', movieId);
      } else {
        await likesAPI.deleteByTarget('Movie', movieId);
      }
    } catch (e) {
      console.error('Frontend: Error toggling like:', e);
      setLiked(previousState); // Revert on error
    } finally {
      setPending(false);
    }
  };

  return (
      <div className="flex items-center gap-3">
        <button
            onClick={handleBack}
            className="px-3 py-2 rounded-lg transition text-sm btn-neutral"
        >
          뒤로가기
        </button>

        <button
            onClick={toggleLike}
            disabled={pending}
            className={`px-3 py-2 rounded-lg transition text-sm flex items-center gap-2 ${
                liked
                    ? 'bg-red-600 hover:bg-red-700 text-white border border-red-600'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700'
            } disabled:opacity-50`}
            aria-pressed={liked}
        >
          <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill={liked ? 'currentColor' : 'none'}
              stroke="currentColor"
              className={`w-5 h-5 ${liked ? 'animate-pulse-once' : ''}`}
          >
            <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.737-.715-1.611-2.377-2.737-4.313-2.737C5.099 3.75 3 5.765 3 8.25c0 7.125 9 12 9 12s9-4.875 9-12z"
            />
          </svg>
          {liked ? '좋아요 취소' : '좋아요'}
        </button>
      </div>
  );
}