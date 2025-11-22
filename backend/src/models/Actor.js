const mongoose = require('mongoose');

const actorSchema = new mongoose.Schema({
  tmdbId: {
    type: Number,
    required: true,
    unique: true
  },
  name: {
    type: String,
    required: true
  },
  nameEnglish: {
    type: String
  },
  profileUrl: {
    type: String
  },
  likeCount: {
    type: Number,
    default: 0
  },
  movieIds: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Movie'
  }]
}, {
  timestamps: true
});

// Create indexes
// actorSchema.index({ tmdbId: 1 }, { unique: true });
actorSchema.index({ name: 'text', nameEnglish: 'text' });

module.exports = mongoose.model('Actor', actorSchema);
