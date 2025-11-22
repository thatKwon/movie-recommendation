const mongoose = require('mongoose');

/**
 * Connect to MongoDB database
 * Uses connection string from environment variable MONGODB_URI
 */
const connectDatabase = async () => {
  try {
    const mongoUri = process.env.MONGODB_URI || process.env.MONGO_URI;
    if (!mongoUri) {
      throw new Error('Missing MONGO_URI (or MONGODB_URI) environment variable');
    }

    const conn = await mongoose.connect(mongoUri, {
      // useNewUrlParser: true,
      // useUnifiedTopology: true
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);

    // Handle connection events
    mongoose.connection.on('error', (err) => {
      console.error('MongoDB connection error:', err);
    });

    mongoose.connection.on('disconnected', () => {
      console.log('MongoDB disconnected');
    });

    // Handle process termination
    process.on('SIGINT', async () => {
      await mongoose.connection.close();
      console.log('MongoDB connection closed due to application termination');
      process.exit(0);
    });

    return conn;
  } catch (error) {
    console.error('Error connecting to MongoDB:', error.message);
    process.exit(1);
  }
};

module.exports = connectDatabase;
