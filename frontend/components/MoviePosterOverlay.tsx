'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { likesAPI } from '@/lib/api';

type Props = {
  movieId: string;
  rating?: string;
  initiallyLiked?: boolean;
  onLikeChange?: (liked: boolean) => void;
};

export default function MoviePosterOverlay({
  movieId, 
  rating, 
  initiallyLiked = false,
  onLikeChange 
}: Props) {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [liked, setLiked] = useState<boolean>(initiallyLiked);
  const [pending, setPending] = useState<boolean>(false);

  const toggleLike = async (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent navigation when clicking like button

    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    if (pending) return;
    setPending(true);
    try {
      if (!liked) {
        console.log('Frontend: Liking Movie. Type: Movie, ID:', movieId);
        await likesAPI.create('Movie', movieId);
        setLiked(true);
        onLikeChange?.(true);
      } else {
        console.log('Frontend: Unliking Movie. Type: Movie, ID:', movieId);
        await likesAPI.deleteByTarget('Movie', movieId);
        setLiked(false);
        onLikeChange?.(false);
      }
    } catch (e) {
      console.error('Frontend: Error toggling like:', e);
      // ignore errors for now
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="absolute top-2 right-2 flex items-center gap-2 z-10">
      {/* Rating */}
      {rating && (
        <div className="bg-black/70 backdrop-blur-sm text-white px-2 py-1 rounded text-xs font-semibold">
          {rating}
        </div>
      )}
      
      {/* Like Button */}
      <button
        onClick={toggleLike}
        disabled={pending}
        className={`p-1.5 rounded-full transition ${
          liked 
            ? 'text-white'
            : 'bg-black/70 backdrop-blur-sm hover:bg-black/90 text-white'
        } disabled:opacity-50`}
        style={{
          backgroundColor: liked ? '#dc2626' : '',
        }}
        onMouseOver={(e) => {
          if (liked) {
            e.currentTarget.style.backgroundColor = '#c52222';
          }
        }}
        onMouseOut={(e) => {
          if (liked) {
            e.currentTarget.style.backgroundColor = '#dc2626';
          }
        }}
        aria-label={liked ? '좋아요 취소' : '좋아요'}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill={liked ? '#dc2626' : 'none'}
          stroke="currentColor"
          strokeWidth={2}
          className="w-4 h-4"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.737-.715-1.611-2.377-2.737-4.313-2.737C5.099 3.75 3 5.765 3 8.25c0 7.125 9 12 9 12s9-4.875 9-12z"
          />
        </svg>
      </button>
    </div>
  );
}
