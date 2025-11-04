export interface Movie {
  id: string;
  title: string;
  titleKo: string;
  description: string;
  year: number;
  director: string;
  actors: string[];
  genres: string[];
  poster: string;
  backdrop: string;
}

export interface CarouselSection {
  id: string;
  title: string;
  movies: Movie[];
}
