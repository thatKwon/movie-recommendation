const mongoose = require('mongoose');

const searchSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  query: {
    type: String,
    required: true
  },
  resultMovieIds: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Movie'
  }],
  searchedAt: {
    type: Date,
    default: Date.now
  }
});

// Create indexes
searchSchema.index({ userId: 1 });
searchSchema.index({ searchedAt: -1 });

module.exports = mongoose.model('Search', searchSchema);
