const axios = require('axios');
const Movie = require('../models/Movie');
const Actor = require('../models/Actor');
const Director = require('../models/Director');
const { TMDB_CONFIG, getImageUrl } = require('../config/tmdb');

/**
 * Get movie by TMDB ID
 * Checks database first, fetches from TMDB API if not cached
 */
const getMovieByTmdbId = async (tmdbId) => {
  try {
    // Check if movie exists in database
    let movie = await Movie.findOne({ tmdbId });

    // If movie exists and was recently fetched (within 30 days), return it
    if (movie) {
      const daysSinceLastFetch = (Date.now() - movie.tmdbLastFetched) / (1000 * 60 * 60 * 24);
      if (daysSinceLastFetch < 30) {
        return movie;
      }
    }

    // Fetch from TMDB API
    const movieUrl = `${TMDB_CONFIG.baseUrl}/movie/${tmdbId}`;
    const creditsUrl = `${TMDB_CONFIG.baseUrl}/movie/${tmdbId}/credits`;

    const [movieResponse, creditsResponse] = await Promise.all([
      axios.get(movieUrl, {
        params: {
          api_key: TMDB_CONFIG.apiKey,
          language: TMDB_CONFIG.language.korean
        }
      }),
      axios.get(creditsUrl, {
        params: {
          api_key: TMDB_CONFIG.apiKey
        }
      })
    ]);

    const movieData = movieResponse.data;
    const creditsData = creditsResponse.data;

    // --- Process Actors ---
    const castArray = [];
    for (let i = 0; i < Math.min(creditsData.cast.length, 15); i++) {
      const castMember = creditsData.cast[i];

      // FIX: Check if path exists before generating URL to prevent broken images
      const profilePath = castMember.profile_path
          ? getImageUrl(castMember.profile_path, 'w185')
          : null;

      // Find or create actor
      let actor = await Actor.findOne({ tmdbId: castMember.id });
      if (!actor) {
        actor = new Actor({
          tmdbId: castMember.id,
          name: castMember.name,
          nameEnglish: castMember.original_name, // Use original_name for English/Native field
          profileUrl: profilePath
        });
        await actor.save();
      }

      castArray.push({
        actorId: actor._id,
        actorName: castMember.name,
        character: castMember.character,
        profileUrl: profilePath, // Save image URL to movie document directly
        order: castMember.order
      });
    }

    // --- Process Directors ---
    const directorsArray = [];
    const directors = creditsData.crew.filter(person => person.job === 'Director');

    for (const directorData of directors) {
      // FIX: Check if path exists
      const profilePath = directorData.profile_path
          ? getImageUrl(directorData.profile_path, 'w185')
          : null;

      // Find or create director
      let director = await Director.findOne({ tmdbId: directorData.id });
      if (!director) {
        director = new Director({
          tmdbId: directorData.id,
          name: directorData.name,
          nameEnglish: directorData.original_name,
          profileUrl: profilePath
        });
        await director.save();
      }

      directorsArray.push({
        directorId: director._id,
        directorName: directorData.name,
        profileUrl: profilePath
      });
    }

    // Prepare movie data
    const movieDocument = {
      tmdbId: movieData.id,
      title: movieData.title,
      titleEnglish: movieData.original_title,
      year: movieData.release_date ? new Date(movieData.release_date).getFullYear() : null,
      genres: movieData.genres.map(g => g.name),
      genresEnglish: movieData.genres.map(g => g.name),
      plot: movieData.overview,
      plotEnglish: movieData.overview,
      // FIX: Check for null paths
      posterUrl: movieData.poster_path ? getImageUrl(movieData.poster_path, 'w500') : null,
      backdropUrl: movieData.backdrop_path ? getImageUrl(movieData.backdrop_path, 'w1280') : null,
      runtime: movieData.runtime,
      rating: movieData.adult ? '청소년 관람불가' : '전체 관람가',
      voteAverage: movieData.vote_average, // Ensure rating is saved
      cast: castArray,
      directors: directorsArray,
      tmdbLastFetched: new Date()
    };

    // Create or update movie
    if (movie) {
      movie = await Movie.findOneAndUpdate(
          { tmdbId },
          movieDocument,
          { new: true }
      );
    } else {
      movie = new Movie(movieDocument);
      await movie.save();

      // Update relations (Optimized for bulk update)
      const actorIds = castArray.map(c => c.actorId);
      if (actorIds.length > 0) {
        await Actor.updateMany(
            { _id: { $in: actorIds } },
            { $addToSet: { movieIds: movie._id } }
        );
      }

      const directorIds = directorsArray.map(d => d.directorId);
      if (directorIds.length > 0) {
        await Director.updateMany(
            { _id: { $in: directorIds } },
            { $addToSet: { movieIds: movie._id } }
        );
      }
    }

    return movie;
  } catch (error) {
    console.error('Error fetching movie from TMDB:', error.message);
    throw new Error('Failed to fetch movie data');
  }
};

/**
 * Search movies on TMDB
 */
const searchMoviesOnTMDB = async (query, filters = {}) => {
  try {
    const response = await axios.get(`${TMDB_CONFIG.baseUrl}/search/movie`, {
      params: {
        api_key: TMDB_CONFIG.apiKey,
        query: query,
        language: TMDB_CONFIG.language.korean,
        page: 1
      }
    });

    const results = response.data.results.slice(0, 20);

    const movies = [];
    for (const result of results) {
      try {
        const movie = await getMovieByTmdbId(result.id);
        movies.push(movie);
      } catch (error) {
        console.error(`Failed to cache movie ${result.id}:`, error.message);
      }
    }

    return movies;
  } catch (error) {
    console.error('Error searching TMDB:', error.message);
    throw new Error('Failed to search movies');
  }
};

/**
 * Get trending movies from TMDB
 */
const getTrendingMovies = async (timeWindow = 'week') => {
  try {
    const response = await axios.get(`${TMDB_CONFIG.baseUrl}/trending/movie/${timeWindow}`, {
      params: {
        api_key: TMDB_CONFIG.apiKey,
        language: TMDB_CONFIG.language.korean
      }
    });

    const results = response.data.results.slice(0, 20);

    const movies = [];
    for (const result of results) {
      try {
        const movie = await getMovieByTmdbId(result.id);
        movies.push(movie);
      } catch (error) {
        console.error(`Failed to cache trending movie ${result.id}:`, error.message);
      }
    }

    return movies;
  } catch (error) {
    console.error('Error fetching trending movies:', error.message);
    return [];
  }
};

module.exports = {
  getMovieByTmdbId,
  searchMoviesOnTMDB,
  getTrendingMovies
};