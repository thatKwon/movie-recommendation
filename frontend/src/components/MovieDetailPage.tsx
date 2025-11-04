import { Movie } from '../types/movie';
import { Heart, ArrowLeft, Calendar, Film as FilmIcon } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface MovieDetailPageProps {
  movie: Movie;
  onBack: () => void;
  isFavoriteMovie: boolean;
  onToggleFavoriteMovie: () => void;
  favoriteDirectors: string[];
  onToggleFavoriteDirector: (director: string) => void;
  favoriteActors: string[];
  onToggleFavoriteActor: (actor: string) => void;
  darkMode?: boolean;
}

export function MovieDetailPage({ 
  movie, 
  onBack,
  isFavoriteMovie,
  onToggleFavoriteMovie,
  favoriteDirectors,
  onToggleFavoriteDirector,
  favoriteActors,
  onToggleFavoriteActor,
  darkMode = true
}: MovieDetailPageProps) {
  return (
    <div className={`min-h-screen ${darkMode ? 'bg-black' : 'bg-zinc-100'}`}>
      {/* Hero Section */}
      <div className="relative h-[50vh] md:h-[60vh]">
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{
            backgroundImage: `url(${movie.backdrop})`,
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/70 to-black/30" />
        
        <div className="relative h-full">
          <Button
            onClick={onBack}
            variant="ghost"
            className="absolute top-20 left-4 md:left-8 text-white hover:bg-white/10"
          >
            <ArrowLeft className="h-5 w-5 mr-2" />
            뒤로가기
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 md:px-8 -mt-32 relative z-10 pb-12">
        <div className="flex flex-col md:flex-row gap-8">
          {/* Poster */}
          <div className="flex-shrink-0">
            <div className="w-48 md:w-64 aspect-[2/3] rounded-lg overflow-hidden shadow-2xl">
              <ImageWithFallback
                src={movie.poster}
                alt={movie.titleKo}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Info */}
          <div className="flex-1 space-y-6">
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h1 className="text-white text-4xl mb-2">{movie.titleKo}</h1>
                  <p className="text-zinc-400 text-xl">{movie.title}</p>
                </div>
                <Button
                  onClick={onToggleFavoriteMovie}
                  size="lg"
                  variant={isFavoriteMovie ? "default" : "outline"}
                  className={isFavoriteMovie ? "bg-red-600 hover:bg-red-700" : "border-zinc-700 hover:bg-zinc-800"}
                >
                  <Heart className={`h-5 w-5 mr-2 ${isFavoriteMovie ? 'fill-current' : ''}`} />
                  {isFavoriteMovie ? '좋아요' : '좋아요'}
                </Button>
              </div>

              <div className="flex flex-wrap items-center gap-4 text-zinc-300">
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  <span>{movie.year}</span>
                </div>
                <div className="flex gap-2">
                  {movie.genres.map(genre => (
                    <Badge key={genre} variant="secondary" className="bg-zinc-800 text-zinc-300">
                      {genre}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <h2 className="text-white text-xl">줄거리</h2>
              <p className="text-zinc-300 leading-relaxed">{movie.description}</p>
            </div>

            <div className="space-y-4">
              {/* Director */}
              <div className="bg-zinc-900 rounded-lg p-6">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <FilmIcon className="h-5 w-5 text-red-600" />
                    <h3 className="text-white">감독</h3>
                  </div>
                  <Button
                    onClick={() => onToggleFavoriteDirector(movie.director)}
                    size="sm"
                    variant="ghost"
                    className="text-zinc-400 hover:text-white"
                  >
                    <Heart className={`h-4 w-4 mr-2 ${favoriteDirectors.includes(movie.director) ? 'fill-red-500 text-red-500' : ''}`} />
                    {favoriteDirectors.includes(movie.director) ? '좋아요' : '좋아요'}
                  </Button>
                </div>
                <p className="text-zinc-300">{movie.director}</p>
              </div>

              {/* Actors */}
              <div className="bg-zinc-900 rounded-lg p-6">
                <div className="flex items-center gap-3 mb-4">
                  <FilmIcon className="h-5 w-5 text-red-600" />
                  <h3 className="text-white">출연</h3>
                </div>
                <div className="space-y-3">
                  {movie.actors.map(actor => (
                    <div key={actor} className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
                      <span className="text-zinc-300">{actor}</span>
                      <Button
                        onClick={() => onToggleFavoriteActor(actor)}
                        size="sm"
                        variant="ghost"
                        className="text-zinc-400 hover:text-white"
                      >
                        <Heart className={`h-4 w-4 ${favoriteActors.includes(actor) ? 'fill-red-500 text-red-500' : ''}`} />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
