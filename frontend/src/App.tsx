import { useState, useEffect } from 'react';
import { LoginPage } from './components/LoginPage';
import { Navigation } from './components/Navigation';
import { MainPage } from './components/MainPage';
import { SearchPage } from './components/SearchPage';
import { UserPage } from './components/UserPage';
import { MovieDetailPage } from './components/MovieDetailPage';
import { Movie, CarouselSection } from './types/movie';
import { initialCarousels } from './data/mockMovies';

type Page = 'home' | 'search' | 'user' | 'detail';

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState<Page>('home');
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);
  const [carousels, setCarousels] = useState<CarouselSection[]>(initialCarousels);
  const [favoriteMovies, setFavoriteMovies] = useState<string[]>([]);
  const [favoriteDirectors, setFavoriteDirectors] = useState<string[]>([]);
  const [favoriteActors, setFavoriteActors] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(true);

  // Check login status from localStorage
  useEffect(() => {
    const loginStatus = localStorage.getItem('isLoggedIn');
    if (loginStatus === 'true') {
      setIsLoggedIn(true);
    }
  }, []);

  // Load favorites from localStorage
  useEffect(() => {
    const savedFavoriteMovies = localStorage.getItem('favoriteMovies');
    const savedFavoriteDirectors = localStorage.getItem('favoriteDirectors');
    const savedFavoriteActors = localStorage.getItem('favoriteActors');
    
    if (savedFavoriteMovies) setFavoriteMovies(JSON.parse(savedFavoriteMovies));
    if (savedFavoriteDirectors) setFavoriteDirectors(JSON.parse(savedFavoriteDirectors));
    if (savedFavoriteActors) setFavoriteActors(JSON.parse(savedFavoriteActors));
  }, []);

  // Save favorites to localStorage
  useEffect(() => {
    localStorage.setItem('favoriteMovies', JSON.stringify(favoriteMovies));
  }, [favoriteMovies]);

  useEffect(() => {
    localStorage.setItem('favoriteDirectors', JSON.stringify(favoriteDirectors));
  }, [favoriteDirectors]);

  useEffect(() => {
    localStorage.setItem('favoriteActors', JSON.stringify(favoriteActors));
  }, [favoriteActors]);

  const handleSelectMovie = (movie: Movie) => {
    setSelectedMovie(movie);
    setCurrentPage('detail');
  };

  const handleNavigate = (page: string) => {
    setCurrentPage(page as Page);
    if (page !== 'detail') {
      setSelectedMovie(null);
    }
  };

  const handleToggleFavoriteMovie = (movieId: string) => {
    setFavoriteMovies(prev => 
      prev.includes(movieId) 
        ? prev.filter(id => id !== movieId)
        : [...prev, movieId]
    );
  };

  const handleToggleFavoriteDirector = (director: string) => {
    setFavoriteDirectors(prev => 
      prev.includes(director) 
        ? prev.filter(d => d !== director)
        : [...prev, director]
    );
  };

  const handleToggleFavoriteActor = (actor: string) => {
    setFavoriteActors(prev => 
      prev.includes(actor) 
        ? prev.filter(a => a !== actor)
        : [...prev, actor]
    );
  };

  const handleLogin = () => {
    setIsLoggedIn(true);
    localStorage.setItem('isLoggedIn', 'true');
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    localStorage.setItem('isLoggedIn', 'false');
    setCurrentPage('home');
  };

  // Show login page if not logged in
  if (!isLoggedIn) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className={darkMode ? 'dark' : ''}>
      <div className={`min-h-screen ${darkMode ? 'bg-black' : 'bg-zinc-100'}`}>
        {currentPage !== 'detail' && (
          <Navigation 
            currentPage={currentPage} 
            onNavigate={handleNavigate} 
            onLogout={handleLogout}
            darkMode={darkMode} 
          />
        )}

        {currentPage === 'home' && (
          <div className="pt-16">
            <MainPage
              carousels={carousels}
              onSelectMovie={handleSelectMovie}
              onUpdateCarousels={setCarousels}
              favoriteMovies={favoriteMovies}
              onToggleFavorite={handleToggleFavoriteMovie}
              darkMode={darkMode}
            />
          </div>
        )}

        {currentPage === 'search' && (
          <SearchPage
            onSelectMovie={handleSelectMovie}
            favoriteMovies={favoriteMovies}
            onToggleFavorite={handleToggleFavoriteMovie}
            darkMode={darkMode}
          />
        )}

        {currentPage === 'user' && (
          <UserPage
            onSelectMovie={handleSelectMovie}
            favoriteMovies={favoriteMovies}
            favoriteDirectors={favoriteDirectors}
            favoriteActors={favoriteActors}
            onToggleFavorite={handleToggleFavoriteMovie}
            onToggleFavoriteDirector={handleToggleFavoriteDirector}
            onToggleFavoriteActor={handleToggleFavoriteActor}
            darkMode={darkMode}
            onToggleDarkMode={() => setDarkMode(!darkMode)}
          />
        )}

        {currentPage === 'detail' && selectedMovie && (
          <div className="pt-16">
            <MovieDetailPage
              movie={selectedMovie}
              onBack={() => setCurrentPage('home')}
              isFavoriteMovie={favoriteMovies.includes(selectedMovie.id)}
              onToggleFavoriteMovie={() => handleToggleFavoriteMovie(selectedMovie.id)}
              favoriteDirectors={favoriteDirectors}
              onToggleFavoriteDirector={handleToggleFavoriteDirector}
              favoriteActors={favoriteActors}
              onToggleFavoriteActor={handleToggleFavoriteActor}
              darkMode={darkMode}
            />
          </div>
        )}
      </div>
    </div>
  );
}
