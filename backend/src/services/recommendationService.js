const axios = require('axios');
const User = require('../models/User');
const Movie = require('../models/Movie');
const Actor = require('../models/Actor');
const Director = require('../models/Director');
const Like = require('../models/Like');
const Click = require('../models/Click');
const { getTrendingMovies } = require('./tmdbService');
const { ServiceUnavailableError } = require('../utils/errors');
const { searchMoviesOnTMDB } = require('./tmdbService');

/**
 * Gathers a user's interaction history.
 * @param {string} userId - The user's MongoDB ObjectId.
 * @returns {Promise<object>} An object containing user history arrays.
 */
const getUserHistory = async (userId) => {
  if (!userId) {
    return {
      likedMovies: [],
      likedActors: [],
      likedDirectors: [],
      clickedMovies: [],
    };
  }

  // Perform queries in parallel for better performance
  const [likes, recentClicks] = await Promise.all([
    Like.find({ userId }).populate({
      path: 'targetId',
      // Dynamically select the model based on the targetType of each document
      model: function(doc) {
        return doc.targetType;
      }
    }),
    Click.find({ userId }).sort({ clickedAt: -1 }).limit(50).populate('movieId')
  ]);

  // Filter and map the results, ensuring targetId is not null
  const likedMovies = likes.filter(like => like.targetType === 'Movie' && like.targetId?.tmdbId).map(like => like.targetId.tmdbId);
  const likedActors = likes.filter(like => like.targetType === 'Actor' && like.targetId?.tmdbId).map(like => like.targetId.tmdbId);
  const likedDirectors = likes.filter(like => like.targetType === 'Director' && like.targetId?.tmdbId).map(like => like.targetId.tmdbId);

  const clickedMovies = recentClicks.map(click => click.movieId?.tmdbId).filter(id => id);

  return { likedMovies, likedActors, likedDirectors, clickedMovies };
};

/**
 * Search with recommendation algorithm
 * @param {string} userId - User's MongoDB ObjectId
 * @param {string} query - Natural language search query
 * @param {object} userPreferences - Optional user preferences
 * @returns {Promise<Array>} Array of recommended movies
 */
const searchWithRecommendation = async (userId, query, userPreferences = {}) => {
  try {
    const user = await User.findById(userId);
    const userHistory = await getUserHistory(userId);

    const prefs = {
      genres: userPreferences.genres?.length ? userPreferences.genres : (user?.preferredGenres || []),
      actors: userPreferences.actors?.length ? userPreferences.actors : (user?.preferredActors || []),
      years: userPreferences.years ? userPreferences.years : (user?.preferredYears || { min: 1990, max: 2024 })
    };

    const requestPayload = {
      user_id: userId, // Add user_id to the payload
      query,
      userPreferences: prefs,
      userHistory
    };

    console.log('--- Recommendation Service ---');
    console.log('User ID:', userId);
    console.log('Liked Movies (TMDB IDs):', userHistory.likedMovies);
    console.log('Liked Actors (TMDB IDs):', userHistory.likedActors);
    console.log('Liked Directors (TMDB IDs):', userHistory.likedDirectors);
    console.log('Preferred Genres:', prefs.genres);
    console.log('-----------------------------');

    if (process.env.RECOMMENDATION_API_URL) {
      try {
        const response = await axios.post(
          `${process.env.RECOMMENDATION_API_URL}/recommend`,
          requestPayload,
          {
            headers: { 'Authorization': `Bearer ${process.env.RECOMMENDATION_API_KEY || ''}`, 'Content-Type': 'application/json' },
            timeout: 10000
          }
        );

        if (response.data?.movies) {
          const recommendedItems = response.data.movies;
          const tmdbIds = recommendedItems.map(item => item.tmdbId);
          const scoreMap = new Map(recommendedItems.map(item => [item.tmdbId, item.score || 0]));

          // Fetch all movies in a single, efficient query
          const foundMovies = await Movie.find({ tmdbId: { $in: tmdbIds } });

          // Map scores and preserve the order from the recommendation service
          const movies = tmdbIds.map(id => {
            const movie = foundMovies.find(m => m.tmdbId === id);
            return movie ? { ...movie.toObject(), relevanceScore: scoreMap.get(id) } : null;
          }).filter(Boolean); // Filter out any nulls if a movie wasn't found

          return movies;
        }
      } catch (error) {
        // Enhanced error logging
        console.error('Recommendation /recommend call failed', {
          url: `${process.env.RECOMMENDATION_API_URL}/recommend`,
          status: error.response?.status,
          data: error.response?.data,
          message: error.message
        });
        // Do not throw an error, instead, fall back to a standard search.
        console.log('Falling back to standard database search.');
      }
    }

    // Fallback logic: Perform a standard text search if the recommendation service fails or is disabled.
    const dbQuery = { $text: { $search: query } };
    if (prefs.genres.length > 0) dbQuery.genres = { $in: prefs.genres };

    let fallbackMovies = await Movie.find(dbQuery)
      .sort({ score: { $meta: 'textScore' }, likeCount: -1, viewCount: -1 })
      .limit(20);

    // If still no results, try TMDB as a last resort
    if (fallbackMovies.length === 0) {
      fallbackMovies = await searchMoviesOnTMDB(query);
    }

    return fallbackMovies.map(movie => ({ ...(movie.toObject ? movie.toObject() : movie), relevanceScore: 0.5 }));

  } catch (error) {
    console.error('Search with recommendation error:', error);
    if (error instanceof ServiceUnavailableError) throw error;
    return [];
  }
};

const PRIMARY_GENRES = [
  '액션', '코미디', '드라마', '판타지', '공포', '로맨스', 'SF', '애니메이션', '가족'
];

/**
 * Fetches top movies for a given genre.
 * @param {string} genre - The genre to fetch movies for.
 * @param {number} limit - The maximum number of movies to return.
 * @returns {Promise<Array>} A promise that resolves to an array of movie documents.
 */
const getMoviesByGenre = async (genre, limit = 20) => {
  return Movie.find({ genres: genre })
    .sort({ likeCount: -1, viewCount: -1, releaseDate: -1 })
    .limit(limit);
};

/**
 * Creates an ordered list of genre-based movie sections.
 * @param {Array<string>} userPreferredGenres - The user's preferred genres.
 * @returns {Promise<Array>} A promise that resolves to an array of section objects.
 */
const createGenreSections = async (userPreferredGenres = []) => {
  const sections = [];
  // Use a Set to ensure genres are unique, then convert back to an array.
  const genresToFetch = [...new Set(userPreferredGenres)];

  // Execute all genre queries in parallel for better performance.
  const moviePromises = genresToFetch.map(genre => getMoviesByGenre(genre));
  const movieResults = await Promise.all(moviePromises);

  // Create a section for each genre that has movies.
  movieResults.forEach((movies, index) => {
    if (movies && movies.length > 0) {
      const genre = genresToFetch[index];
      sections.push({ title: `${genre} 추천`, movies });
    }
  });

  return sections;
};

/**
 * Get home page recommendations
 * @param {string|object} params - userId string or { userId, userPreferences }
 * @returns {Promise<Array>} Array of recommendation sections
 */
const getHomeRecommendations = async (params) => {
  try {
    const resolvedUserId = typeof params === 'string' ? params : params?.userId;
    const explicitPrefs = typeof params === 'object' ? (params.userPreferences || {}) : {};
    const user = resolvedUserId ? await User.findById(resolvedUserId) : null;

    let sections = [];

    // --- Section 1: New Movies ---
    const newMovies = await Movie.find().sort({ releaseDate: -1 }).limit(20);
    if (newMovies.length > 0) {
      sections.push({ title: '새로운 영화', movies: newMovies });
    }

    // --- Section 2: User's Preferred Genre Carousels ---
    // Create sections for the genres the user has explicitly liked.
    const preferredGenres = user?.preferredGenres || [];
    if (preferredGenres.length > 0) {
      console.log(`Generating sections for user's preferred genres: ${preferredGenres.join(', ')}`);
      // Fetch movies for preferred genres in parallel.
      const preferredGenreSections = await createGenreSections(preferredGenres);
      sections.push(...preferredGenreSections);
    }

    // --- Section 3: Carousels for Other Primary Genres ---
    // Create sections for the rest of the genres from the questionnaire list.
    const otherPrimaryGenres = PRIMARY_GENRES.filter(g => !preferredGenres.includes(g));
    const otherGenreSections = await createGenreSections(otherPrimaryGenres);
    sections.push(...otherGenreSections);

    // --- Section 4 (Optional): Personalized rows from Python service ---
    // This can be added back if you want ML-based rows like "Similar to what you watched"
    // For now, the focus is on genre carousels as requested.

    if (sections.length === 0) {
      // Fallback if no sections could be generated at all
      console.log('No sections generated, falling back to trending movies.');
      const trendingMovies = await getTrendingMovies();
      if (trendingMovies.length > 0) {
        sections.push({ title: '지금 뜨는 인기 영화', movies: trendingMovies });
      }
    }

    return sections;
  } catch (error) {
    console.error('Home recommendations error:', error.message);
    throw error;
  }
};
/**
 * Get all items liked by a user, with full details populated.
 * @param {string} userId The user's MongoDB ObjectId.
 * @returns {Promise<Array>} A promise that resolves to an array of populated Like documents.
 */
const getPopulatedUserLikes = async (userId) => {
  if (!userId) {
    return [];
  }
  const userLikes = await Like.find({ userId })
    .sort({ createdAt: -1 }) // Show most recently liked items first
    .populate({
      path: 'targetId',
      model: function(doc) {
        return doc.targetType;
      },
    });

  // Filter out any likes where the underlying movie/actor/director may have been deleted
  return userLikes.filter(like => like.targetId);
};

/**
 * Toggles a like on a target item (Movie, Actor, Director) for a user.
 * If the like exists, it's removed (unlike). If it doesn't, it's created (like).
 * Also updates the likeCount on the target model.
 * @param {string} userId The user's MongoDB ObjectId.
 * @param {string} targetId The MongoDB ObjectId of the target item.
 * @param {'Movie' | 'Actor' | 'Director'} targetType The type of the target item.
 * @returns {Promise<{isLiked: boolean, likeCount: number}>} An object indicating the new like status and the updated like count.
 */
const toggleLike = async (userId, targetId, targetType) => {
  const modelMap = {
    Movie,
    Actor,
    Director,
  };

  const TargetModel = modelMap[targetType];
  if (!TargetModel) {
    throw new Error('Invalid target type specified.');
  }

  // 1. Check if the like already exists
  const existingLike = await Like.findOne({ userId, targetId, targetType });

  let isLiked;
  let updateOperation;

  if (existingLike) {
    // 2a. If it exists, remove it (unlike)
    await Like.findByIdAndDelete(existingLike._id);
    updateOperation = { $inc: { likeCount: -1 } };
    isLiked = false;
  } else {
    // 2b. If it doesn't exist, create it (like)
    await Like.create({ userId, targetId, targetType });
    updateOperation = { $inc: { likeCount: 1 } };
    isLiked = true;
  }

  // 3. Update the likeCount on the target document (Movie, Actor, etc.)
  const updatedTarget = await TargetModel.findByIdAndUpdate(
    targetId,
    updateOperation,
    { new: true } // Return the updated document
  );

  return { isLiked, likeCount: updatedTarget.likeCount };
};

module.exports = {
  searchWithRecommendation,
  getHomeRecommendations,
  getPopulatedUserLikes,
  toggleLike,
};
