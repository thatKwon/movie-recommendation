'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { likesAPI } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import MoviePosterOverlay from '@/components/MoviePosterOverlay';

type Movie = {
  id: string;
  title: string;
  year?: number;
  posterUrl?: string;
  rating?: string;
};

export default function ActorMoviesGrid({ movies }: { movies: Movie[] }) {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [likedMap, setLikedMap] = useState<Record<string, boolean>>({});

  const items = useMemo(() => (movies || []).map(m => ({ type: 'Movie', id: String(m.id) })), [movies]);

  useEffect(() => {
    if (!isAuthenticated || items.length === 0) return;
    likesAPI
      .check(items)
      .then((res) => {
        const map: Record<string, boolean> = {};
        const liked = res.data?.liked || {};
        for (const it of items) {
          map[it.id] = Boolean(liked[`Movie_${it.id}`]);
        }
        setLikedMap(map);
      })
      .catch(() => {});
  }, [isAuthenticated, items]);

  const sorted = useMemo(() => {
    const copy = [...(movies || [])];
    copy.sort((a, b) => {
      const la = likedMap[String(a.id)] ? 1 : 0;
      const lb = likedMap[String(b.id)] ? 1 : 0;
      if (lb !== la) return lb - la; // liked first
      // fallback: newer year first
      const ya = a.year ? Number(a.year) : 0;
      const yb = b.year ? Number(b.year) : 0;
      return yb - ya;
    });
    return copy;
  }, [movies, likedMap]);

  if (!movies || movies.length === 0) {
    return <div className="text-gray-400">표시할 작품이 없습니다</div>;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
      {sorted.map((m) => (
        <div key={m.id} className="group cursor-pointer" onClick={() => router.push(`/movie/${m.id}`)}>
          <div className="relative aspect-[2/3] bg-gray-800 rounded-lg overflow-hidden mb-2 group-hover:ring-2 group-hover:ring-red-500 transition">
            {m.posterUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={m.posterUrl} alt={m.title} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-500">No Image</div>
            )}
            {/* Rating and Like Button Overlay */}
            <MoviePosterOverlay
              movieId={String(m.id)}
              rating={m.rating}
              initiallyLiked={likedMap[String(m.id)] || false}
              onLikeChange={(liked) => {
                setLikedMap(prev => ({ ...prev, [String(m.id)]: liked }));
              }}
            />
          </div>
          <div className="text-sm font-medium line-clamp-2">{m.title}</div>
          <div className="text-xs text-gray-400">{m.year}</div>
        </div>
      ))}
    </div>
  );
}
