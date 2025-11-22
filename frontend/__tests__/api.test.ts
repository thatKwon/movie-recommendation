// Define the mock implementation for the axios instance first
const mockApiClient = {
    get: jest.fn(),
    post: jest.fn(),
    patch: jest.fn(),
    delete: jest.fn(),
    interceptors: {
        request: { use: jest.fn(), eject: jest.fn() },
        response: { use: jest.fn(), eject: jest.fn() },
    },
};

// Use the factory parameter of jest.mock to provide a custom module implementation
jest.mock('axios', () => ({
    // We need to keep the actual 'isAxiosError' function, otherwise parts of the app might break
    ...jest.requireActual('axios'),
    // Mock the 'create' function to return our mock client
    create: jest.fn(() => mockApiClient),
}));

// Now, we can safely import the api module. It will be created using the mocked axios.create.
import { moviesAPI } from '../lib/api';
import axios from 'axios';

describe('moviesAPI', () => {
    beforeEach(() => {
        // Clear the history of all mocks before each test
        jest.clearAllMocks();
    });

    it('getMovie should fetch and return a movie', async () => {
        const mockMovieData = { id: '1', title: 'Inception' };
        // Configure the mock 'get' function to return a successful response
        (mockApiClient.get as jest.Mock).mockResolvedValue({ data: { movie: mockMovieData } });

        // Call the API function we are testing
        const result = await moviesAPI.getMovie('1');

        // Assert that the function returned the expected data
        expect(result).toEqual({ movie: mockMovieData });
        // Assert that the mock 'get' function was called with the correct URL
        expect(mockApiClient.get).toHaveBeenCalledWith('/api/movies/1');
    });
});
