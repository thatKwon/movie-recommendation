import { Movie } from '../types/movie';
import { MovieCard } from './MovieCard';
import { Moon, Sun, Heart, User as UserIcon, Film } from 'lucide-react';
import { Button } from './ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { mockMovies } from '../data/mockMovies';

interface UserPageProps {
  onSelectMovie: (movie: Movie) => void;
  favoriteMovies: string[];
  favoriteDirectors: string[];
  favoriteActors: string[];
  onToggleFavorite: (movieId: string) => void;
  onToggleFavoriteDirector: (director: string) => void;
  onToggleFavoriteActor: (actor: string) => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

export function UserPage({ 
  onSelectMovie,
  favoriteMovies,
  favoriteDirectors,
  favoriteActors,
  onToggleFavorite,
  onToggleFavoriteDirector,
  onToggleFavoriteActor,
  darkMode,
  onToggleDarkMode
}: UserPageProps) {
  const likedMovies = mockMovies.filter(m => favoriteMovies.includes(m.id));
  
  // 모든 감독 목록 추출
  const allDirectors = Array.from(new Set(mockMovies.map(m => m.director)));
  
  // 모든 배우 목록 추출
  const allActors = Array.from(new Set(mockMovies.flatMap(m => m.actors)));

  return (
    <div className={`min-h-screen pt-24 pb-12 px-4 md:px-8 ${darkMode ? 'bg-black' : 'bg-zinc-100'}`}>
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Profile Section */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-6">
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-red-600 to-red-800 flex items-center justify-center">
              <UserIcon className="w-12 h-12 text-white" />
            </div>
            <div className="space-y-2">
              <h1 className={`text-3xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>마이페이지</h1>
              <p className={darkMode ? 'text-zinc-400' : 'text-zinc-600'}>좋아요한 영화, 감독, 배우를 관리하세요</p>
            </div>
          </div>

          <Button
            onClick={onToggleDarkMode}
            variant="outline"
            className={darkMode 
              ? 'bg-zinc-900 border-zinc-800 hover:bg-zinc-800 text-white' 
              : 'bg-white border-zinc-300 hover:bg-zinc-50 text-zinc-900'
            }
          >
            {darkMode ? (
              <>
                <Sun className="h-5 w-5 mr-2" />
                라이트 모드
              </>
            ) : (
              <>
                <Moon className="h-5 w-5 mr-2" />
                다크 모드
              </>
            )}
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className={`rounded-lg p-6 space-y-2 ${
            darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-200'
          }`}>
            <Film className="h-8 w-8 text-red-600" />
            <div className={`text-2xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>{favoriteMovies.length}</div>
            <div className={`text-sm ${darkMode ? 'text-zinc-400' : 'text-zinc-600'}`}>좋아요한 영화</div>
          </div>
          <div className={`rounded-lg p-6 space-y-2 ${
            darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-200'
          }`}>
            <UserIcon className="h-8 w-8 text-red-600" />
            <div className={`text-2xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>{favoriteDirectors.length}</div>
            <div className={`text-sm ${darkMode ? 'text-zinc-400' : 'text-zinc-600'}`}>좋아요한 감독</div>
          </div>
          <div className={`rounded-lg p-6 space-y-2 ${
            darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-200'
          }`}>
            <Heart className="h-8 w-8 text-red-600" />
            <div className={`text-2xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>{favoriteActors.length}</div>
            <div className={`text-sm ${darkMode ? 'text-zinc-400' : 'text-zinc-600'}`}>좋아요한 배우</div>
          </div>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="movies" className="space-y-6">
          <TabsList className={darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-300'}>
            <TabsTrigger value="movies">영화</TabsTrigger>
            <TabsTrigger value="directors">감독</TabsTrigger>
            <TabsTrigger value="actors">배우</TabsTrigger>
          </TabsList>

          <TabsContent value="movies" className="space-y-4">
            {likedMovies.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {likedMovies.map((movie) => (
                  <MovieCard
                    key={movie.id}
                    movie={movie}
                    onSelect={onSelectMovie}
                    isFavorite={true}
                    onToggleFavorite={onToggleFavorite}
                  />
                ))}
              </div>
            ) : (
              <div className={`text-center py-16 ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
                아직 좋아요한 영화가 없습니다
              </div>
            )}
          </TabsContent>

          <TabsContent value="directors" className="space-y-4">
            {favoriteDirectors.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {allDirectors.filter(d => favoriteDirectors.includes(d)).map((director) => {
                  const directorMovies = mockMovies.filter(m => m.director === director);
                  return (
                    <div key={director} className={`rounded-lg p-6 space-y-3 ${
                      darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-200'
                    }`}>
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className={`text-lg ${darkMode ? 'text-white' : 'text-zinc-900'}`}>{director}</h3>
                          <p className={`text-sm ${darkMode ? 'text-zinc-400' : 'text-zinc-600'}`}>{directorMovies.length}편의 작품</p>
                        </div>
                        <button
                          onClick={() => onToggleFavoriteDirector(director)}
                          className={`p-2 rounded-full transition-colors ${
                            darkMode ? 'hover:bg-zinc-800' : 'hover:bg-zinc-100'
                          }`}
                        >
                          <Heart className="h-5 w-5 fill-red-500 text-red-500" />
                        </button>
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {directorMovies.slice(0, 3).map(movie => (
                          <span key={movie.id} className={`text-xs ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
                            {movie.titleKo}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className={`text-center py-16 ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
                아직 좋아요한 감독이 없습니다
              </div>
            )}
          </TabsContent>

          <TabsContent value="actors" className="space-y-4">
            {favoriteActors.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {allActors.filter(a => favoriteActors.includes(a)).map((actor) => {
                  const actorMovies = mockMovies.filter(m => m.actors.includes(actor));
                  return (
                    <div key={actor} className={`rounded-lg p-6 space-y-3 ${
                      darkMode ? 'bg-zinc-900' : 'bg-white border border-zinc-200'
                    }`}>
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className={`text-lg ${darkMode ? 'text-white' : 'text-zinc-900'}`}>{actor}</h3>
                          <p className={`text-sm ${darkMode ? 'text-zinc-400' : 'text-zinc-600'}`}>{actorMovies.length}편 출연</p>
                        </div>
                        <button
                          onClick={() => onToggleFavoriteActor(actor)}
                          className={`p-2 rounded-full transition-colors ${
                            darkMode ? 'hover:bg-zinc-800' : 'hover:bg-zinc-100'
                          }`}
                        >
                          <Heart className="h-5 w-5 fill-red-500 text-red-500" />
                        </button>
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {actorMovies.slice(0, 3).map(movie => (
                          <span key={movie.id} className={`text-xs ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
                            {movie.titleKo}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className={`text-center py-16 ${darkMode ? 'text-zinc-500' : 'text-zinc-400'}`}>
                아직 좋아요한 배우가 없습니다
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
