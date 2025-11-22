'use client';

import { use, useEffect, useState } from 'react';
import { notFound, useRouter } from 'next/navigation'; // Added useRouter
import { directorsAPI } from '@/lib/api';
import Loading from '@/components/Loading';
import Error from '@/components/Error';
import MoviePoster from '@/components/MoviePoster';

type Director = {
  id: string;
  name: string;
  nameEnglish: string;
  profileUrl: string;
  movies: Movie[];
};

type Movie = {
  id: string;
  title: string;
  year: number;
  posterUrl: string;
};

type Props = {
  params: Promise<{ id: string }>;
};

export default function DirectorPage({ params }: Props) {
  const { id } = use(params);
  const router = useRouter(); // Initialize router

  const [director, setDirector] = useState<Director | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id || id === 'undefined') {
      setLoading(false);
      return;
    }

    const fetchDirector = async () => {
      try {
        const directorData = await directorsAPI.getById(id);
        setDirector(directorData);
      } catch (err: any) {
        if (err.response?.status === 404) {
          notFound();
        } else {
          setError('Failed to fetch director data.');
        }
      } finally {
        setLoading(false);
      }
    };
    fetchDirector();
  }, [id]);

  if (loading) return <Loading />;
  if (error) return <Error message={error} />;
  if (!director) return null;

  return (
      <div className="min-h-screen bg-black text-white px-4 py-8 md:px-8">
        {/* Header Section */}
        <div className="max-w-7xl mx-auto">
          <div className="flex justify-between items-start mb-12 mt-4">
            <div className="flex items-center gap-6">
              {/* Circular Profile Image */}
              <div className="w-24 h-24 md:w-32 md:h-32 rounded-full overflow-hidden bg-gray-800 ring-2 ring-white/20 flex-shrink-0">
                {director.profileUrl ? (
                    <img
                        src={director.profileUrl}
                        alt={director.name}
                        className="w-full h-full object-cover"
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-500 text-sm">No Image</div>
                )}
              </div>

              {/* Director Name */}
              <div>
                <h1 className="text-3xl md:text-4xl font-bold mb-1">{director.name}</h1>
                <p className="text-gray-400 text-sm md:text-base">{director.nameEnglish}</p>
              </div>
            </div>

            {/* Back Button */}
            <button
                onClick={() => router.back()}
                className="px-4 py-1.5 rounded-full border border-gray-600 text-sm text-gray-300 hover:bg-white/10 transition-colors"
            >
              뒤로가기
            </button>
          </div>

          {/* Content Section */}
          <div>
            <h2 className="text-lg font-bold mb-6 text-white">출연작</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-x-4 gap-y-8">
              {director.movies.map((movie) => (
                  <MoviePoster key={movie.id} movie={movie} />
              ))}
            </div>
          </div>
        </div>
      </div>
  );
}