'use client';

import { useRef } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import MoviePosterOverlay from './MoviePosterOverlay';

interface Movie {
  id: number;
  title: string;
  year: string;
  posterUrl?: string;
  rating?: number;
  userLiked?: boolean;
  genres?: string[];
}

interface CarouselProps {
  title: string;
  movies: Movie[];
  loading: boolean;
  onLikeChange: () => void;
  hideEmptyMessage?: boolean;
}

export default function Carousel({ title, movies, loading, onLikeChange, hideEmptyMessage = false }: CarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const handleScroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const scrollAmount = scrollRef.current.offsetWidth * 0.8;
      scrollRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth',
      });
    }
  };

  if (loading) {
    return (
      <section className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-6">{title}</h2>
        <div className="text-center py-8"><p className="text-gray-400">영화를 불러오는 중...</p></div>
      </section>
    );
  }

  if (movies.length === 0) {
    if (hideEmptyMessage) {
      return null;
    }
    return (
      <section className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-6">{title}</h2>
        <div className="text-center py-8"><p className="text-gray-400">{title} 영화가 없습니다</p></div>
      </section>
    );
  }

  return (
    <section className="container mx-auto px-4 py-8">
      <h2 className="text-2xl font-bold mb-6">{title}</h2>
      <div className="relative group">
        <button
          onClick={() => handleScroll('left')}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-20 p-2 bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-black/80"
          aria-label="Scroll Left"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
        </button>
        
        <div ref={scrollRef} className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide">
          {movies.map((movie) => (
            <div key={movie.id} onClick={() => router.push(`/movie/${movie.id}`)} className="flex-shrink-0 w-[180px] cursor-pointer">
              <div className="relative aspect-[2/3] bg-gray-800 rounded-lg overflow-hidden mb-3 hover:ring-2 hover:ring-red-500 transition">
                {movie.posterUrl ? (
                  <Image src={movie.posterUrl} alt={movie.title} width={180} height={270} className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-500 text-sm">No Image</div>
                )}
                <MoviePosterOverlay movieId={movie.id.toString()} rating={movie.rating?.toString()} initiallyLiked={movie.userLiked} onLikeChange={onLikeChange} />
              </div>
              <h3 className="font-medium line-clamp-2 text-sm">{movie.title}</h3>
              <p className="text-xs text-gray-400">{movie.year} {movie.genres?.[0] ? `• ${movie.genres[0]}` : ''}</p>
            </div>
          ))}
        </div>

        <button
          onClick={() => handleScroll('right')}
          className="absolute right-0 top-1/2 -translate-y-1/2 z-20 p-2 bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-black/80"
          aria-label="Scroll Right"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
        </button>
      </div>
    </section>
  );
}
