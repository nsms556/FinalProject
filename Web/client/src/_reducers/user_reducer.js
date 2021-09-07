import {AUTH_USER, LOGIN_USER, LOGOUT_USER, REGISTER_USER} from '../_actions/types';


// eslint-disable-next-line import/no-anonymous-default-export
export default function (state = {}, action) {

    switch (action.type) {
        case AUTH_USER:
            return {...state, userData: action.payload}
        case LOGIN_USER:
            return {...state, loginSuccess: action.payload}
        case LOGOUT_USER:
            return {...state}
        case REGISTER_USER:
            return {...state, register: action.payload}
        default:
            return state;
    }
}
