import { useState } from 'react';
import { Movie } from '../types/movie';
import { MovieCard } from './MovieCard';
import { Search as SearchIcon } from 'lucide-react';
import { Input } from './ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { mockMovies } from '../data/mockMovies';

interface SearchPageProps {
  onSelectMovie: (movie: Movie) => void;
  favoriteMovies: string[];
  onToggleFavorite: (movieId: string) => void;
  darkMode?: boolean;
}

export function SearchPage({ onSelectMovie, favoriteMovies, onToggleFavorite, darkMode = true }: SearchPageProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState<'all' | 'title' | 'director' | 'actor'>('all');

  const filteredMovies = mockMovies.filter(movie => {
    if (!searchQuery.trim()) return false;

    const query = searchQuery.toLowerCase();

    switch (searchType) {
      case 'title':
        return movie.title.toLowerCase().includes(query) || 
               movie.titleKo.toLowerCase().includes(query);
      case 'director':
        return movie.director.toLowerCase().includes(query);
      case 'actor':
        return movie.actors.some(actor => actor.toLowerCase().includes(query));
      case 'all':
      default:
        return movie.title.toLowerCase().includes(query) ||
               movie.titleKo.toLowerCase().includes(query) ||
               movie.director.toLowerCase().includes(query) ||
               movie.actors.some(actor => actor.toLowerCase().includes(query)) ||
               movie.genres.some(genre => genre.toLowerCase().includes(query));
    }
  });

  return (
    <div className={`min-h-screen pt-24 pb-12 px-4 md:px-8 ${darkMode ? 'bg-black' : 'bg-zinc-100'}`}>
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="space-y-4">
          <h1 className={`text-3xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>검색</h1>
          
          <div className="relative">
            <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-400" />
            <Input
              type="text"
              placeholder="영화 제목, 감독, 배우를 검색하세요"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`pl-12 h-14 ${
                darkMode 
                  ? 'bg-zinc-900 border-zinc-800 text-white placeholder:text-zinc-500'
                  : 'bg-white border-zinc-300 text-zinc-900 placeholder:text-zinc-400'
              }`}
            />
          </div>

          <Tabs value={searchType} onValueChange={(v) => setSearchType(v as any)}>
            <TabsList className={darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-300'}>
              <TabsTrigger value="all">전체</TabsTrigger>
              <TabsTrigger value="title">제목</TabsTrigger>
              <TabsTrigger value="director">감독</TabsTrigger>
              <TabsTrigger value="actor">배우</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {searchQuery.trim() ? (
          <>
            <div className={darkMode ? 'text-zinc-400' : 'text-zinc-600'}>
              {filteredMovies.length}개의 결과
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {filteredMovies.map((movie) => (
                <MovieCard
                  key={movie.id}
                  movie={movie}
                  onSelect={onSelectMovie}
                  isFavorite={favoriteMovies.includes(movie.id)}
                  onToggleFavorite={onToggleFavorite}
                />
              ))}
            </div>

            {filteredMovies.length === 0 && (
              <div className={`text-center py-16 ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
                검색 결과가 없습니다
              </div>
            )}
          </>
        ) : (
          <div className={`text-center py-16 ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
            검색어를 입력하세요
          </div>
        )}
      </div>
    </div>
  );
}
