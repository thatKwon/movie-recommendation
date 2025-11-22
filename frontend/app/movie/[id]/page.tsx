'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Image from 'next/image'; // Using Next.js Image for optimization

import { moviesAPI } from '@/lib/api';
import Loading from '@/components/Loading';
import ErrorComponent from '@/components/Error';
import MovieActions from '@/components/MovieActions';
import PersonList from '@/components/PersonList';

type Movie = {
  id: string;
  tmdbId: number;
  title: string;
  titleEnglish: string;
  year: number;
  genres: string[];
  plot: string;
  posterUrl: string;
  backdropUrl: string;
  runtime: number;
  rating: string;
  voteAverage?: number; // Defines the rating field
  cast: any[];
  directors: any[];
  likeCount: number;
  viewCount: number;
  userLiked: boolean;
};

export default function MoviePage() {
  const params = useParams();
  const id = params.id as string;
  const [movie, setMovie] = useState<Movie | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    const fetchMovie = async () => {
      try {
        const movieData = await moviesAPI.getMovie(id);
        setMovie(movieData);
      } catch (err) {
        setError('Failed to fetch movie data.');
      } finally {
        setLoading(false);
      }
    };
    fetchMovie();
  }, [id]);

  if (loading) return <Loading />;
  if (error) return <ErrorComponent message={error} />;
  if (!movie) return null;

  return (
      <div className="min-h-screen bg-black text-white pb-20">
        {/* Backdrop hero */}
        <div
            className="relative w-full h-[40vh] md:h-[55vh]"
            style={{
              backgroundImage: movie?.backdropUrl ? `url(${movie.backdropUrl})` : 'none',
              backgroundSize: 'cover',
              backgroundPosition: 'center top'
            }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-black/40 to-black" />
        </div>

        {/* Content */}
        <section className="container mx-auto px-4 -mt-32 relative z-10">
          <div className="flex flex-col md:flex-row gap-10">

            {/* Left Column: Poster */}
            <div className="flex flex-col gap-4 w-full md:w-72 flex-shrink-0">
              <div className="aspect-[2/3] rounded-xl overflow-hidden shadow-2xl ring-1 ring-white/20 bg-gray-900 relative">
                {movie?.posterUrl ? (
                    <Image
                        src={movie.posterUrl}
                        alt={movie.title}
                        fill
                        className="object-cover"
                        sizes="(max-width: 768px) 100vw, 300px"
                        priority
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-500">No Image</div>
                )}
              </div>
              <div className="flex justify-center">
                <MovieActions movieId={String(movie.id)} initiallyLiked={Boolean(movie.userLiked)} />
              </div>
            </div>

            {/* Right Column: Details */}
            <div className="flex-1 pt-2 md:pt-8">
              <h1 className="text-4xl md:text-5xl font-bold mb-2 leading-tight">{movie.title}</h1>
              {movie.titleEnglish && <p className="text-xl text-gray-400 mb-6 font-light">{movie.titleEnglish}</p>}

              <div className="flex flex-wrap items-center gap-3 text-sm text-gray-300 mb-8">
                {movie.year && <span className="px-3 py-1 bg-gray-800 rounded-full border border-gray-700">{movie.year}</span>}

                {/* --- RATING DISPLAY --- */}
                <span className="px-3 py-1 bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 rounded-full font-medium">
                  ★ {(movie.voteAverage || 0).toFixed(1)} / 10
                </span>
                {/* ---------------------- */}

                {movie.runtime && <span className="px-3 py-1 bg-gray-800 rounded-full border border-gray-700">{movie.runtime}분</span>}

                {Array.isArray(movie.genres) && movie.genres.map((g: string) => (
                    <span key={g} className="px-3 py-1 bg-gray-800 rounded-full border border-gray-700">{g}</span>
                ))}
              </div>

              {movie.plot && (
                  <div className="mb-10">
                    <h2 className="text-lg font-bold text-white mb-3 border-l-4 border-red-600 pl-3">줄거리</h2>
                    <p className="text-gray-300 leading-relaxed text-lg">{movie.plot}</p>
                  </div>
              )}

              {/* Directors */}
              {Array.isArray(movie.directors) && movie.directors.length > 0 && (
                  <div className="mb-10">
                    <h3 className="text-lg font-bold text-white mb-4 border-l-4 border-red-600 pl-3">감독</h3>
                    <PersonList type="Director" items={movie.directors} />
                  </div>
              )}

              {/* Cast */}
              {Array.isArray(movie.cast) && movie.cast.length > 0 && (
                  <div className="mb-10">
                    <h3 className="text-lg font-bold text-white mb-4 border-l-4 border-red-600 pl-3">출연진</h3>
                    <PersonList type="Actor" items={movie.cast.slice(0, 12)} />
                  </div>
              )}
            </div>
          </div>
        </section>
      </div>
  );
}