/* eslint-disable react-hooks/exhaustive-deps */
import React, {useEffect} from "react";
import {useSelector, useDispatch} from "react-redux";

import {auth} from '../_actions/user_actions';


// eslint-disable-next-line import/no-anonymous-default-export
export default function (SpecificComponent, option) {

    function AuthenticationCheck(props) {

        let user = useSelector(state => state.user);
        const dispatch = useDispatch();

        useEffect(() => {
            // To know my current status, send Auth request
            dispatch(auth()).then(_ => {

                const username = localStorage.getItem('username');

                if (!username) {
                    // Not Loggined in Status
                    if (option) {
                        props.history.push('/auth/login');
                    }
                } else if (option === false) {
                    // Loggined in Status
                    props.history.push('/')
                }
            })
        }, [])

        return (
            <SpecificComponent {...props} user={user}/>
        )
    }

    return AuthenticationCheck
}
