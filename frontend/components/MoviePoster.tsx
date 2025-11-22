import Link from 'next/link';
import MoviePosterOverlay from './MoviePosterOverlay';

type Movie = {
    id: string;
    title: string;
    year: number;
    posterUrl: string;
};

type Props = {
    movie: Movie;
};

export default function MoviePoster({ movie }: Props) {
    return (
        <Link href={`/movie/${movie.id}`} className="block group relative rounded-lg overflow-hidden bg-gray-900">
            <div className="aspect-[2/3] w-full relative">
                {movie.posterUrl && (
                    <img
                        src={movie.posterUrl}
                        alt={movie.title}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                )}

                {/* Gradient Overlay for Text Readability */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/40 to-transparent opacity-100" />

                {/* Hover Interaction Overlay */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                <div className="absolute inset-0">
                    <MoviePosterOverlay movieId={movie.id} />
                </div>

                {/* Text Content - Positioned at bottom */}
                <div className="absolute bottom-0 left-0 w-full p-4">
                    <h3 className="font-bold text-white text-lg leading-tight mb-1 truncate">{movie.title}</h3>
                    <p className="text-gray-400 text-sm">{movie.year}</p>
                </div>
            </div>
        </Link>
    );
}