import { useState, useRef } from 'react';
import { Movie } from '../types/movie';
import { MovieCard } from './MovieCard';
import { ChevronLeft, ChevronRight, X, GripVertical } from 'lucide-react';

interface MovieCarouselProps {
  title: string;
  movies: Movie[];
  onSelectMovie: (movie: Movie) => void;
  onDelete?: () => void;
  isDragging?: boolean;
  favoriteMovies?: string[];
  onToggleFavorite?: (movieId: string) => void;
}

export function MovieCarousel({ 
  title, 
  movies, 
  onSelectMovie, 
  onDelete,
  isDragging,
  favoriteMovies = [],
  onToggleFavorite
}: MovieCarouselProps) {
  const [scrollPosition, setScrollPosition] = useState(0);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: 'left' | 'right') => {
    if (!scrollContainerRef.current) return;
    
    const container = scrollContainerRef.current;
    const scrollAmount = container.clientWidth * 0.8;
    const newPosition = direction === 'left' 
      ? Math.max(0, scrollPosition - scrollAmount)
      : Math.min(container.scrollWidth - container.clientWidth, scrollPosition + scrollAmount);
    
    container.scrollTo({ left: newPosition, behavior: 'smooth' });
    setScrollPosition(newPosition);
  };

  const handleScroll = () => {
    if (scrollContainerRef.current) {
      setScrollPosition(scrollContainerRef.current.scrollLeft);
    }
  };

  const showLeftArrow = scrollPosition > 0;
  const showRightArrow = scrollContainerRef.current 
    ? scrollPosition < scrollContainerRef.current.scrollWidth - scrollContainerRef.current.clientWidth - 10
    : true;

  return (
    <div className={`group/carousel relative ${isDragging ? 'opacity-50' : ''}`}>
      <div className="mb-4 flex items-center justify-between px-4 md:px-12">
        <div className="flex items-center gap-3">
          <div className="cursor-grab active:cursor-grabbing">
            <GripVertical className="h-5 w-5 text-zinc-500" />
          </div>
          <h2 className="text-white">{title}</h2>
        </div>
        <button
          onClick={onDelete}
          className="p-2 rounded-full hover:bg-zinc-800 transition-colors"
          aria-label="삭제"
        >
          <X className="h-5 w-5 text-zinc-400 hover:text-white" />
        </button>
      </div>

      <div className="relative group/scroll px-4 md:px-12">
        {showLeftArrow && (
          <button
            onClick={() => scroll('left')}
            className="absolute left-0 top-0 bottom-0 z-10 w-12 md:w-16 bg-gradient-to-r from-black/80 to-transparent flex items-center justify-start pl-2 opacity-0 group-hover/scroll:opacity-100 transition-opacity"
            aria-label="이전"
          >
            <div className="p-2 rounded-full bg-black/50 backdrop-blur-sm hover:bg-black/70 transition-colors">
              <ChevronLeft className="h-6 w-6 text-white" />
            </div>
          </button>
        )}

        <div
          ref={scrollContainerRef}
          onScroll={handleScroll}
          className="flex gap-2 overflow-x-auto scrollbar-hide scroll-smooth"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {movies.map((movie) => (
            <div key={movie.id} className="flex-shrink-0 w-36 sm:w-44 md:w-52">
              <MovieCard
                movie={movie}
                onSelect={onSelectMovie}
                isFavorite={favoriteMovies.includes(movie.id)}
                onToggleFavorite={onToggleFavorite}
              />
            </div>
          ))}
        </div>

        {showRightArrow && (
          <button
            onClick={() => scroll('right')}
            className="absolute right-0 top-0 bottom-0 z-10 w-12 md:w-16 bg-gradient-to-l from-black/80 to-transparent flex items-center justify-end pr-2 opacity-0 group-hover/scroll:opacity-100 transition-opacity"
            aria-label="다음"
          >
            <div className="p-2 rounded-full bg-black/50 backdrop-blur-sm hover:bg-black/70 transition-colors">
              <ChevronRight className="h-6 w-6 text-white" />
            </div>
          </button>
        )}
      </div>
    </div>
  );
}
