import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 10000,
});

export const getKPIs = () => api.get('/api/kpis').then(r => r.data);
export const getSegments = () => api.get('/api/segments').then(r => r.data);
export const getSegmentsRevenue = () => api.get('/api/segments/revenue').then(r => r.data);
export const getSegment = (id) => api.get(`/api/segments/${id}`).then(r => r.data);
export const getCategories = () => api.get('/api/categories').then(r => r.data);
export const getCategory = (id) => api.get(`/api/categories/${id}`).then(r => r.data);
export const getCustomer = (id) => api.get(`/api/customers/${id}`).then(r => r.data);
export const listCustomers = (limit = 50, offset = 0) => api.get('/api/customers', { params: { limit, offset } }).then(r => r.data);
export const predictSimple = (data) => api.post('/api/predict/simple', data).then(r => r.data);
export const predictFromItems = (data) => api.post('/api/predict', data).then(r => r.data);
export const biQuery = (question) => api.post('/api/bi/query', { question }).then(r => r.data);
export const healthCheck = () => api.get('/api/health').then(r => r.data);

export default api;
