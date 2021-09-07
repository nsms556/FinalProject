import axios from "axios";

import {AUTH_USER, LOGIN_USER, LOGOUT_USER, REGISTER_USER} from './types';


export function auth() {

    const request = axios.get('http://127.0.0.1:8000/admin')
        .then(response => response.data);

    return {
        type: AUTH_USER,
        payload: request
    }
}


export function loginUser(dataToSubmit) {

    const request = axios.post('http://127.0.0.1:8000/users/login', dataToSubmit)
        .then(response => response.data);

    return {
        type: LOGIN_USER,
        payload: request
    }
}


export function logoutUser() {

    const request = axios.get('http://127.0.0.1:8000/users/logout')
        .then(response => response.data);

    return {
        type: LOGOUT_USER,
        payload: request
    }
}


export function registerUser(dataToSubmit) {

    const request = axios.post('http://127.0.0.1:8000/users/register', dataToSubmit)
        .then(response => response.data);

    return {
        type: REGISTER_USER,
        payload: request
    }
}
