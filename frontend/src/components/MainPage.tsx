import { useState, useRef, useEffect } from 'react';
import { Movie, CarouselSection } from '../types/movie';
import { MovieCarousel } from './MovieCarousel';
import { Search, Sparkles } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { mockMovies } from '../data/mockMovies';
import { motion, AnimatePresence } from 'motion/react';

interface MainPageProps {
  carousels: CarouselSection[];
  onSelectMovie: (movie: Movie) => void;
  onUpdateCarousels: (carousels: CarouselSection[]) => void;
  favoriteMovies: string[];
  onToggleFavorite: (movieId: string) => void;
  darkMode?: boolean;
}

export function MainPage({ 
  carousels, 
  onSelectMovie, 
  onUpdateCarousels,
  favoriteMovies,
  onToggleFavorite,
  darkMode = true
}: MainPageProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [newPersonalCarouselId, setNewPersonalCarouselId] = useState<string | null>(null);
  const [deletingCarouselId, setDeletingCarouselId] = useState<string | null>(null);
  const dragCounter = useRef(0);
  const autoScrollInterval = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (newPersonalCarouselId) {
      const timer = setTimeout(() => setNewPersonalCarouselId(null), 500);
      return () => clearTimeout(timer);
    }
  }, [newPersonalCarouselId]);

  // Cleanup auto-scroll on unmount
  useEffect(() => {
    return () => {
      if (autoScrollInterval.current) {
        clearInterval(autoScrollInterval.current);
      }
    };
  }, []);

  const handleSearch = () => {
    if (!searchQuery.trim()) return;

    // Create placeholder movies for the search result
    const placeholderMovies = Array.from({ length: 10 }, (_, i) => ({
      id: `search-${Date.now()}-${i + 1}`,
      title: `Movie ${i + 1}`,
      titleKo: `영화 ${i + 1}`,
      description: 'Placeholder movie description',
      year: 2024,
      director: 'Director',
      actors: ['Actor 1', 'Actor 2'],
      genres: ['Genre'],
      poster: '',
      backdrop: ''
    }));

    const personalCarousel: CarouselSection = {
      id: 'personal-' + Date.now(),
      title: `"${searchQuery}" - 지금 당신을 위한 영화`,
      movies: placeholderMovies
    };

    setNewPersonalCarouselId(personalCarousel.id);

    // Always add new carousel at the beginning, never replace
    onUpdateCarousels([personalCarousel, ...carousels]);
    setSearchQuery('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragEnter = (index: number) => {
    dragCounter.current++;
    if (draggedIndex !== null && draggedIndex !== index) {
      setDragOverIndex(index);
    }
  };

  const handleDragLeave = () => {
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setDragOverIndex(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    
    // Auto-scroll based on mouse position
    const scrollThreshold = 100; // pixels from top/bottom to trigger scroll
    const scrollSpeed = 10;
    const mouseY = e.clientY;
    const windowHeight = window.innerHeight;
    
    // Clear existing interval
    if (autoScrollInterval.current) {
      clearInterval(autoScrollInterval.current);
      autoScrollInterval.current = null;
    }
    
    // Scroll down when near bottom
    if (mouseY > windowHeight - scrollThreshold) {
      autoScrollInterval.current = setInterval(() => {
        window.scrollBy({ top: scrollSpeed, behavior: 'auto' });
      }, 16);
    }
    // Scroll up when near top
    else if (mouseY < scrollThreshold) {
      autoScrollInterval.current = setInterval(() => {
        window.scrollBy({ top: -scrollSpeed, behavior: 'auto' });
      }, 16);
    }
  };

  const handleDrop = (index: number, e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    
    // Clear auto-scroll interval
    if (autoScrollInterval.current) {
      clearInterval(autoScrollInterval.current);
      autoScrollInterval.current = null;
    }
    
    if (draggedIndex === null || draggedIndex === index) {
      setDraggedIndex(null);
      setDragOverIndex(null);
      return;
    }

    const newCarousels = [...carousels];
    const [draggedItem] = newCarousels.splice(draggedIndex, 1);
    newCarousels.splice(index, 0, draggedItem);
    
    onUpdateCarousels(newCarousels);
    setDraggedIndex(null);
    setDragOverIndex(null);
  };
  
  const handleDragEnd = () => {
    // Clear auto-scroll interval when drag ends
    if (autoScrollInterval.current) {
      clearInterval(autoScrollInterval.current);
      autoScrollInterval.current = null;
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
    dragCounter.current = 0;
  };

  const handleDeleteCarousel = (carouselId: string) => {
    setDeletingCarouselId(carouselId);
    // Wait for animation to complete before actually removing
    setTimeout(() => {
      onUpdateCarousels(carousels.filter(c => c.id !== carouselId));
      setDeletingCarouselId(null);
    }, 300);
  };

  return (
    <div ref={containerRef} className={`min-h-screen pb-12 ${darkMode ? 'bg-black' : 'bg-zinc-100'}`}>
      {/* Hero Section */}
      <div className="relative h-[60vh] md:h-[70vh] mb-8">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-zinc-900"
          style={{
            backgroundImage: mockMovies[0]?.backdrop ? `url(${mockMovies[0].backdrop})` : 'none',
          }}
        />
        <div className={`absolute inset-0 bg-gradient-to-t ${
          darkMode 
            ? 'from-black via-black/50 to-transparent' 
            : 'from-zinc-100 via-zinc-100/50 to-transparent'
        }`} />
        
        <div className="relative h-full flex flex-col items-center justify-center px-4">
          <div className="w-full max-w-3xl space-y-6">
            <div className="text-center space-y-2">
              <h1 className={`text-3xl md:text-4xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>당신을 위한 완벽한 영화를 찾아드립니다</h1>
              <p className={darkMode ? 'text-zinc-300' : 'text-zinc-700'}>지금 기분이나 보고 싶은 영화를 설명해보세요</p>
            </div>
            
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-400" />
                <Input
                  type="text"
                  placeholder="예: 우주를 배경으로 한 감동적인 영화가 보고 싶어요"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className={`pl-12 h-14 backdrop-blur-sm ${
                    darkMode 
                      ? 'bg-white/10 border-white/20 text-white placeholder:text-zinc-400 focus:bg-white/20'
                      : 'bg-white/70 border-zinc-300 text-zinc-900 placeholder:text-zinc-500 focus:bg-white'
                  }`}
                />
              </div>
              <Button
                onClick={handleSearch}
                className="h-14 px-6 bg-red-600 hover:bg-red-700"
              >
                <Sparkles className="h-5 w-5 mr-2" />
                <span className="hidden sm:inline">추천받기</span>
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Carousels */}
      <div className="space-y-8 md:space-y-12">
        <AnimatePresence mode="popLayout">
          {carousels.map((carousel, index) => {
            const isNewPersonalCarousel = carousel.id === newPersonalCarouselId;
            const isDeleting = carousel.id === deletingCarouselId;
            const isDragging = draggedIndex === index;
            
            return (
              <motion.div
                key={carousel.id}
                draggable
                onDragStart={() => handleDragStart(index)}
                onDragEnter={() => handleDragEnter(index)}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(index, e)}
                onDragEnd={handleDragEnd}
                layout
                initial={isNewPersonalCarousel ? { opacity: 0, y: -20 } : false}
                animate={{
                  opacity: isDeleting ? 0 : isDragging ? 0.5 : 1,
                  scale: isDeleting ? 0.8 : 1,
                  y: isDragging ? -10 : dragOverIndex !== null && dragOverIndex === index ? 60 : 0,
                  filter: isDragging ? 'grayscale(50%)' : 'grayscale(0%)'
                }}
                exit={{
                  opacity: 0,
                  scale: 0.8,
                  transition: { duration: 0.3 }
                }}
                transition={{
                  layout: { duration: 0.3, ease: "easeInOut" },
                  opacity: { duration: isDeleting ? 0.3 : 0.2 },
                  scale: { duration: isDeleting ? 0.3 : 0.2 },
                  y: { duration: 0.3, ease: "easeOut" },
                  filter: { duration: 0.2 }
                }}
              >
                <MovieCarousel
                  title={carousel.title}
                  movies={carousel.movies}
                  onSelectMovie={onSelectMovie}
                  onDelete={() => handleDeleteCarousel(carousel.id)}
                  isDragging={isDragging}
                  favoriteMovies={favoriteMovies}
                  onToggleFavorite={onToggleFavorite}
                />
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
