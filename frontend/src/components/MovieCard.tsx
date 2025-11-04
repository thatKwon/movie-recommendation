import { Movie } from '../types/movie';
import { Heart } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface MovieCardProps {
  movie: Movie;
  onSelect: (movie: Movie) => void;
  isFavorite?: boolean;
  onToggleFavorite?: (movieId: string) => void;
}

export function MovieCard({ movie, onSelect, isFavorite, onToggleFavorite }: MovieCardProps) {
  const isPlaceholder = !movie.poster;
  
  return (
    <div 
      className="group relative cursor-pointer transition-transform duration-300 hover:scale-105"
      onClick={() => onSelect(movie)}
    >
      <div className="relative aspect-[2/3] overflow-hidden rounded-lg bg-zinc-800">
        {isPlaceholder ? (
          <div className="h-full w-full bg-gradient-to-br from-zinc-700 to-zinc-800" />
        ) : (
          <ImageWithFallback
            src={movie.poster}
            alt={movie.titleKo}
            className="h-full w-full object-cover"
          />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
        
        {onToggleFavorite && !isPlaceholder && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleFavorite(movie.id);
            }}
            className="absolute top-2 right-2 z-10 p-2 rounded-full bg-black/50 backdrop-blur-sm opacity-0 transition-opacity duration-300 group-hover:opacity-100 hover:bg-black/70"
          >
            <Heart 
              className={`h-5 w-5 ${isFavorite ? 'fill-red-500 text-red-500' : 'text-white'}`}
            />
          </button>
        )}
        
        {!isPlaceholder && (
          <div className="absolute bottom-0 left-0 right-0 p-4 translate-y-2 opacity-0 transition-all duration-300 group-hover:translate-y-0 group-hover:opacity-100">
            <h3 className="mb-1 text-white">{movie.titleKo}</h3>
            <div className="flex items-center gap-2 text-sm text-zinc-300">
              <span>{movie.year}</span>
              <span>•</span>
              <span>{movie.genres[0]}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
