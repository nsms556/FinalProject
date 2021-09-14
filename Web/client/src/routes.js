/*!

=========================================================
* Argon Dashboard React - v1.2.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-dashboard-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-dashboard-react/blob/master/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

import Index from "views/Index.js";
import Profile from "views/examples/Profile.js";
import Register from "views/examples/Register.js";
import Login from "views/examples/Login.js";

import PlaylistPage from "views/PlaylistPage/PlaylistPage";
import LogoutPage from "views/LogoutPage/LogoutPage";
import SearchPage from "views/SearchPage/SearchPage";

import Auth from "./hoc/auth";


const username = localStorage.getItem('username');


let routes = [
    {
        path: "/index",
        name: "HOME/MAIN",
        icon: "ni ni-tv-2 text-black",
        component: Auth(Index, null),
        layout: "/admin",
    },
    {
        path: "/search",
        name: "Search",
        icon: "ni ni-archive-2 text-primary",
        component: Auth(SearchPage, null),
        layout: "/admin",
    },
    {
        path: "/playlist",
        name: "Playlist",
        icon: "ni ni-headphones text-danger",
        component: Auth(PlaylistPage, true),
        layout: "/admin",
    },
];


if (username) {
    routes.push(
        {
            path: "/user-profile",
            name: "User Profile",
            icon: "ni ni-single-02 text-yellow",
            component: Auth(Profile, true),
            layout: "/admin",
        },
        {
            path: "/logout",
            name: "Logout",
            icon: "ni ni-button-power text-default",
            component: Auth(LogoutPage, true),
            layout: "/auth",
        }
    );
} else {
    routes.push(
        {
            path: "/login",
            name: "Login",
            icon: "ni ni-key-25 text-info",
            component: Auth(Login, false),
            layout: "/auth",
        },
        {
            path: "/register",
            name: "Register",
            icon: "ni ni-circle-08 text-gray",
            component: Auth(Register, false),
            layout: "/auth",
        }
    );
}

export default routes;
